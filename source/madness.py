import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise, GaussianDropout
from keras.utils.np_utils import to_categorical
from keras.regularizers import WeightRegularizer, ActivityRegularizer, l2, activity_l2
from keras.optimizers import Adam
from sklearn.preprocessing import normalize

# Original data comes from Kaggle's March Madness competition.
# Game-level data for regular seasons and tournaments back to 1985.
# More detailed data only exists back to 2003. We'll use that.
# https://www.kaggle.com/c/march-machine-learning-mania-2017/data
def load_data():
	prefix = "../data/"
	files = ["RegularSeasonDetailedResults.csv", "Seasons.csv", "Teams.csv", "TourneyDetailedResults.csv", "TourneySeeds.csv", "TourneySlots.csv"]
	dfs = []
	for file in files:
		df = pd.read_csv(prefix + file)
		dfs.append(df)

	return dfs # Return list of pandas DataFrames for all data files.

# This function does a lot of the heavy lifting. We convert game-level data
# into season averages for each team in each year. Then we calculate more
# advanced metrics like RPI, offensive and defensive efficiency, true shooting,
# etc.
# This function has a lot of operations that could have been mapped to functions.
# A lot of repetition. -10 elegance points.
def calc_stats(results, teams, seeds, years, a):
	rows = []

	# Need to iterate through each year in our dataset and handle it separately.
	# 2015 Oregon stats are not taken into account for Oregon's 2016 team, etc.
	for year in years:
		print("Processing " + str(year) + " data...")
		row = []
		year_results = results[results["Season"] == year] # Grab just games from the current year.
		year_seeds = seeds[seeds["Season"] == year] # Grab tournament seeds for current year.
		
		# Now that we've picked a year, iterate through every team that made the
		# tournament that year.
		for team in year_seeds["Team"]:

			# Grab the current team's games for the current year, then add in it's tournament seed.
			year_team = teams[teams["Team_Id"] == team]
			year_team = year_team.set_index("Team_Id").join(year_seeds.set_index("Team"))
			tid = year_team.index.values[0]
			s = year_team["Seed"].values[0]
			year_team["Seed"] = int(s[1:3])
			
			# Because of the way the data is structured, it's easiest to
			# break the games into ones lost by team in question and ones won.
			win_games = year_results[year_results["Wteam"] == tid]
			loss_games = year_results[year_results["Lteam"] == tid]
			year_games = pd.concat([win_games, loss_games])
			dn_win = win_games["Daynum"]
			dn_loss = loss_games["Daynum"]
			sum_win = dn_win.sum()
			sum_loss = dn_loss.sum()
			
			# Undefeated teams throw divide-by-0 errors.
			if sum_loss == 0:
				sum_loss = 0.00001
			wins = win_games.shape[0]

			# Tally wins and losses.
			losses = loss_games.shape[0]
			played = wins + losses
			year_team["Wins"] = wins
			year_team["Losses"] = losses

			# Split wins and losses by location. This is necessary to calculate RPI.
			home_wins = win_games[win_games["Wloc"] == "H"].shape[0]
			home_losses = loss_games[loss_games["Wloc"] == "A"].shape[0]
			away_wins = win_games[win_games["Wloc"] == "A"].shape[0]
			away_losses = loss_games[loss_games["Wloc"] == "H"].shape[0]
			neut_wins = win_games[win_games["Wloc"] == "N"].shape[0]
			neut_losses = loss_games[loss_games["Wloc"] == "N"].shape[0]

			# Now we start the RPI calculation. Two things here:
			# 	- I didn't do the full calculation. Only got around to RPI
			#	  the morning the tournament started so I didn't do the last step,
			#	  opponent's opponent's weighted win%
			#	- I definitely should have broken this off into its own function.
			#	  -10 elegance points
			
			# RPI considers away wins and home losses more important than the opposite.
			weighted_wins = (0.6 * home_wins) + neut_wins + (1.4 * away_wins)
			weighted_losses = (0.6 * away_losses) + neut_losses + (1.4 * home_losses)
			weighted_percent = float(weighted_wins) / (weighted_wins + weighted_losses)

			# Compile a list of every team our current team played this season.
			opponents = []
			for i, game in year_games.iterrows():
				if game["Wteam"] == tid:
					opp = game["Lteam"]
				else:
					opp = game["Wteam"]
				opponents.append(opp)

			# Calculate weighted win% for each opponent.
			opp_weighted_wins = []
			opp_weighted_losses = []
			for opp in opponents:
				opp_wins = year_results[year_results["Wteam"] == opp]
				opp_losses = year_results[year_results["Lteam"] == opp]
				opp_h_w = opp_wins[opp_wins["Wloc"] == "H"].shape[0]
				opp_h_l = opp_losses[opp_losses["Wloc"] == "A"].shape[0]
				opp_a_w = opp_wins[opp_wins["Wloc"] == "A"].shape[0]
				opp_a_l = opp_losses[opp_losses["Wloc"] == "H"].shape[0]
				opp_n_w = opp_wins[opp_wins["Wloc"] == "N"].shape[0]
				opp_n_l = opp_losses[opp_losses["Wloc"] == "N"].shape[0]

				opp_weighted_wins.append((0.6 * opp_h_w) + opp_n_w + (1.4 * opp_a_w))
				opp_weighted_losses.append((0.6 * opp_a_l) + opp_n_l + (1.4 * opp_n_l))

			opp_ww = sum(opp_weighted_wins) / float(len(opp_weighted_wins))
			opp_wl = sum(opp_weighted_losses) / float(len(opp_weighted_losses))
			owp = opp_ww / (opp_ww + opp_wl)
			
			# My RPI-lite: 1/3 * WP + 2/3 * OWP
			# Real RPI: 1/4 * WP + 1/2 * OWP + 1/4 * OOWP
			year_team["RPI"] = ((float(1)/3.0) * weighted_percent) + ((float(2)/3.0) * owp)

			win_points = (win_games["Wscore"] * dn_win).sum() / sum_win
			loss_points = (loss_games["Wscore"] * dn_loss).sum() / sum_loss
			ppg = (win_points + loss_points) / played
			
			win_against = (win_games["Lscore"] * dn_win).sum() / sum_win
			loss_against = (loss_games["Wscore"] * dn_loss).sum() / sum_loss
			ppga = (win_against + loss_against) / played

			# Field goals made and attempted
			win_fgm = (win_games["Wfgm"] * dn_win).sum() / sum_win
			loss_fgm = (loss_games["Lfgm"] * dn_loss).sum() / sum_loss
			win_fga = (win_games["Wfga"] * dn_win).sum() / sum_win
			loss_fga = (loss_games["Lfga"] * dn_loss).sum() / sum_loss
			fgm = win_fgm + loss_fgm
			fga = win_fga + loss_fga

			# 3-point field goals made and attempted.
			win_3pm = (win_games["Wfgm3"] * dn_win).sum() / sum_win
			loss_3pm = (loss_games["Lfgm3"] * dn_loss).sum() / sum_loss
			win_3pa = (win_games["Wfga3"] * dn_win).sum() / sum_win
			loss_3pa = (loss_games["Lfga3"] * dn_loss).sum() / sum_loss
			treys = win_3pm + loss_3pm
			treys_a = win_3pa + loss_3pa

			# Opposing team field goals made and attempted.
			win_fgma = (win_games["Lfgm"] * dn_win).sum() / sum_win
			loss_fgma = (loss_games["Wfgm"] * dn_loss).sum() / sum_loss
			win_fgaa = (win_games["Lfga"] * dn_win).sum() / sum_win
			loss_fgaa = (loss_games["Wfga"] * dn_loss).sum() / sum_loss			
			fgma = win_fgma + loss_fgma
			fgaa = win_fgaa + loss_fgaa

			# Opposing team 3-point field goals made and attempted.
			win_3pma = (win_games["Lfgm3"] * dn_win).sum() / sum_win
			loss_3pma = (loss_games["Wfgm3"] * dn_loss).sum() / sum_loss
			win_3paa = (win_games["Lfga3"] * dn_win).sum() / sum_win
			loss_3paa = (loss_games["Wfga3"] * dn_loss).sum() / sum_loss			
			treysa = win_3pma + loss_3pma
			treys_aa = win_3paa + loss_3paa

			# Free throws made and attempted
			win_ftm = (win_games["Wftm"] * dn_win).sum() / sum_win
			loss_ftm = (loss_games["Lftm"] * dn_loss).sum() / sum_loss
			win_fta = (win_games["Wfta"] * dn_win).sum() / sum_win
			loss_fta = (loss_games["Lfta"] * dn_loss).sum() / sum_loss
			ftm = win_ftm + loss_ftm
			fta = win_fta + loss_fta
			year_team["FT%"] = float(ftm) / fta

			# Opposing team free throws made and attempted.
			win_ftma = (win_games["Lftm"] * dn_win).sum() / sum_win
			loss_ftma = (loss_games["Wftm"] * dn_loss).sum() / sum_loss
			win_ftaa = (win_games["Lfta"] * dn_win).sum() / sum_win
			loss_ftaa = (loss_games["Wfta"] * dn_loss).sum() / sum_loss
			ftma = win_ftma + loss_ftma
			ftaa = win_ftaa + loss_ftaa

			# Offensive rebounds gotten vs. defensive rebounds given.
			win_or = (win_games["Wor"] * dn_win).sum() / sum_win
			loss_or = (loss_games["Lor"] * dn_loss).sum() / sum_loss
			win_dra = (win_games["Ldr"] * dn_win).sum() / sum_win
			loss_dra = (loss_games["Wdr"] * dn_loss).sum() / sum_loss
			or_for = win_or + loss_or
			dra = win_dra + loss_dra
			offensive_glass = float(or_for) / (or_for + dra)
			year_team["Offensive Rebounding"] = offensive_glass

			# Defensive rebounds gotten vs. offensive rebounds given.
			win_dr = (win_games["Wdr"] * dn_win).sum() / sum_win
			loss_dr = (loss_games["Ldr"] * dn_loss).sum() / sum_loss
			win_ora = (win_games["Lor"] * dn_win).sum() / sum_win
			loss_ora = (loss_games["Wor"] * dn_loss).sum() / sum_loss
			dr_for = win_dr + loss_dr
			ora = win_ora + loss_ora
			defensive_glass = float(dr_for) / (dr_for + ora)
			year_team["Defensive Rebounding"] = defensive_glass

			# Assist ratio.
			win_ass = (win_games["Wast"] * dn_win).sum() / sum_win
			loss_ass = (loss_games["Last"] * dn_loss).sum() / sum_loss
			win_assa = (win_games["Last"] * dn_win).sum() / sum_win
			loss_assa = (loss_games["Wast"] * dn_loss).sum() / sum_loss
			ass_for = win_ass + loss_ass
			ass_against = win_assa + loss_assa
			ass_ratio = float(ass_for - ass_against) / ass_against
			year_team["Assist Ratio"] = ass_ratio

			# Turnover ratio.	
			win_to = (win_games["Wto"] * dn_win).sum() / sum_win
			loss_to = (loss_games["Lto"] * dn_loss).sum() / sum_loss
			win_toa = (win_games["Lto"] * dn_win).sum() / sum_win
			loss_toa = (loss_games["Wto"] * dn_loss).sum() / sum_loss
			to_for = win_to + loss_to
			to_against = win_toa + loss_toa
			to_ratio = float(to_for - to_against) / to_against
			year_team["T/O Ratio"] = to_ratio

			# Fouls.
			win_pf = (win_games["Wpf"] * dn_win).sum() / sum_win
			loss_pf = (loss_games["Lpf"] * dn_loss).sum() / sum_loss
			win_pfa = (win_games["Lpf"] * dn_win).sum() / sum_win
			loss_pfa = (loss_games["Wpf"] * dn_loss).sum() / sum_loss
			pf = win_pf + loss_pf
			pfa = win_pfa + loss_pfa
			pfpg = float(pf) / played
			pfapg = float(pfa) / played
			year_team["Fouls/Game"] = pfpg
			year_team["Fouls Against/Game"] = pfapg

			# Calculate offensive and defensive efficiency.
			poss = .96 * (fga - or_for + to_for + (.44 * fta))
			off_rating = ppg * 100 / poss
			def_rating = ppga * 100 / poss
			year_team["Off Rating"] = off_rating
			year_team["Def Rating"] = def_rating

			# Calculate effective FG% for and against.
			efg = (fgm + (0.5 * treys)) / fga
			efga = (fgma + (0.5 * treysa)) / fgaa
			year_team["EFG"] = efg
			year_team["EFGA"] = efga

			year_team["Unique"] = year_team["Team_Name"] + str(year)

			row.append(year_team)

		row = pd.concat(row)
		rows.append(row)
	rows = pd.concat(rows)

	return rows

# This function takes team season stats, combines into training vectors,
# and labels them based on winner.
def label(stats, results, years):
	table = []
	labels = []
	for year in years:
		result_year = results[results["Season"] == year]
		stats_year = stats[stats["Season"] == year]
		teams = result_year[["Wteam", "Lteam"]]
		for i, game in teams.iterrows():
			w = game["Wteam"]
			l = game["Lteam"]
			w_team = stats_year.loc[w]
			l_team = stats_year.loc[l]
			w_team = w_team.to_frame()
			l_team = l_team.to_frame()
			w_team = w_team.transpose()
			l_team = l_team.transpose()
			w_team = w_team.values
			l_team = l_team.values

			# Trim off the non-numerical features like team name and season.
			w_team = np.delete(w_team, -1)
			w_team = np.delete(w_team, 0)
			l_team = np.delete(l_team, -1)
			l_team = np.delete(l_team, 0)
			w_team = np.delete(w_team, 0)
			l_team = np.delete(l_team, 0)
			forward = np.append(w_team, l_team)
			backward = np.append(l_team, w_team)
			table.append(forward)
			table.append(backward)

			# Make a vector labeled 1 with winning team first,
			# 0 with losing team first.
			labels.append(1)
			labels.append(0)

	return table, labels

# Secondary labeling function for later on when we want to
# predict actual tournaments.
def label2(t1, t2):
	t1 = t1.to_frame()
	t2 = t2.to_frame()
	t1 = t1.transpose()
	t2 = t2.transpose()
	t1 = t1.values
	t2 = t2.values

	# Non-numerical value trimming.
	t1 = np.delete(t1, -1)
	t2 = np.delete(t2, -1)
	t1 = np.delete(t1, 0)
	t2 = np.delete(t2, 0)
	t1 = np.delete(t1, 0)
	t2 = np.delete(t2, 0)

	vector = np.append(t1, t2)
	return vector

# Compare predicted outcomes against real ones to
# find accuracy of the model.
def accuracy(ys, y):
	correct = 0
	for i in range(len(y)):
		if ys[i] == y[i]:
			correct += 1
	accuracy = float(correct) / len(y)
	print("%.2f" % (accuracy * 100) + "%")

# This is an important one. It uses the structure of the actual bracket
# to iterate through and predict a tournament, rather than predicting
# every possible matchup.
# We have a list of seeds and the later rounds they feed into,
# predict the games we have seeds for, then propagate the winners forward.
def predict_tournament(year, slots, teams, stats, model, lu):
	results = []

	slots = slots[slots["Season"] == year]
	teams = teams[teams["Season"] == year]

	# Initialize a queue of games to predict with first round games.
	seeds = teams["Seed"].tolist()
	for i, game in slots.iterrows():
		if game["Strongseed"] in seeds:
			lookup = teams[teams["Seed"] == game["Strongseed"]]
			team = int(lookup["Team"])
			index = slots.index == i
			slots.loc[index, "Strongseed"] = team
		if game["Weakseed"] in seeds:
			lookup = teams[teams["Seed"] == game["Weakseed"]]
			team = int(lookup["Team"])
			index = slots.index == i
			slots.loc[index, "Weakseed"] = team
	queue = slots

	# Work through our queue until we get to the end. When a game is predicted,
	# the winner's next game is sent to the back of the queue.
	while len(queue) > 1:
		next = queue.head(n=1)
		queue = queue.drop(next.index)
		strongid = int(next["Strongseed"])
		weakid = int(next["Weakseed"])
		strong = stats.loc[strongid]
		weak = stats.loc[weakid]
		X = label2(strong, weak)
		X = normalize(X)
		X = X.reshape(1, -1)

		# This is where we guess the winner!
		y = model.predict_classes(X, batch_size = 1, verbose=0)
		pred = y[0]
		if pred == 1:
			winner = strongid
			loser = weakid
		else:
			winner = weakid
			loser = strongid
		s = next["Slot"].values[0]
		info = [s, winner, loser]
		results.append(info)

		# Change the placeholder corresponding to this game
		# to the winner's seed, update the queue as such.
		if len(queue) == 1:
			replaced = queue.iloc[0]
			replaced["Weakseed"] = winner
			queue.iloc[0] = replaced
		else:
			replaced = queue[queue["Strongseed"] == s]
			if len(replaced) == 0:
				replaced = queue[queue["Weakseed"] == s]
				replaced["Weakseed"] = winner
			else:
				replaced["Strongseed"] = winner
			queue.loc[replaced.index] = replaced
	
	# Typing issues made it much easier to handle the last entry on its own.
	# Mostly the same as above.
	last = queue.head(n=1)
	strongid = int(last["Strongseed"])
	weakid = int(last["Weakseed"])
	strong = stats.loc[strongid]
	weak = stats.loc[weakid]
	X = label2(strong, weak)
	X = normalize(X)
	X = X.reshape(1, -1)
	y = model.predict_classes(X, batch_size=1, verbose=0)
	pred = y[0]
	if pred == 1:
		winner = strongid
		loser = weakid
	else:
		winner = weakid
		loser = strongid
	s = last["Slot"].values[0]
	info = [s, winner, loser]
	results.append(info)

	# Collate results and look up team names to replaced IDs.
	for i in range(len(results)):
		w = lu[lu["Team_Id"] == results[i][1]]
		l = lu[lu["Team_Id"] == results[i][2]]
		results[i][1] = w["Team_Name"].values[0]
		results[i][2] = l["Team_Name"].values[0]

	return results

# Prints out predicted bracket in a (somewhat) readable format.
def display_pred(results):
	print("Play-in games:")
	for i in range(4):
		print(results[i][1] + " over " + results[i][2])

	print("\nFirst round:")
	for j in range(4, 36):
		print(results[j][1] + " over " + results[j][2])

	print("\nSecond round:")
	for k in range(36, 52):
		print(results[k][1] + " over " + results[k][2])

	print("\nSweet 16:")
	for l in range(52, 60):
		print(results[l][1] + " over " + results[l][2])

	print("\nElite 8:")
	for m in range(60, 64):
		print(results[m][1] + " over " + results[m][2])

	print("\nFinal four:")
	for n in range(64, 66):
		print(results[n][1] + " over " + results[n][2])

	print("\nChampionship game:")
	print(results[-1][1] + " over " + results[-1][2])
	print("\nSaving results...")
	labels = ["Slot", "Winner", "Loser"]
	df = pd.DataFrame.from_records(results, columns=labels)
	with open("../data/madness.csv", "a") as f:
		df.to_csv(f, header=False)
	print("Done.")

# This function predicts every possible matchup in a tournament and outputs
# the results to a .csv. Used for Kaggle submission.
def pred_tournaments(years, seeds, model, stats):
	preds = []
	for year in years:
		year_stats = stats[stats["Season"] == year]
		year_seeds = seeds[seeds["Season"] == year]
		year_teams = sorted(year_seeds["Team"].values)
		for i in range(len(year_teams)):
			for j in range(i, len(year_teams)):
				if i != j:
					a = year_stats.loc[year_teams[i]]
					b = year_stats.loc[year_teams[j]]
					a_id = year_teams[i]
					b_id = year_teams[j]
					vector = label2(a, b)
					vector = normalize(vector)
					vector = vector.reshape(1, -1)
					pred = model.predict(vector)
					matchup = str(year) + "_" + str(a_id) + "_" + str(b_id)
					pred = pred[0][1]
					row = [matchup, pred]
					preds.append(row)
	labels = ["Id", "Pred"]
	df = pd.DataFrame.from_records(preds, columns=labels)
	with open("../data/submission.csv", "w") as f:
		df.to_csv(f, index=False)

# This is where we define our network. Pretty fun to tweak things.
def define_model():
	model = Sequential([
		# For each training iteration, we're going to drop 10% of the input
		# nodes, randomly.  This helps avoid overfitting.
	    Dropout(0.1, input_shape=(len(X[0]),)),

	    # We normalize the inputs to make it easier to weight them correctly.
	    BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None),
	    GaussianNoise(0.01), # Just a tiny bit of noise.
	    
	    # Ended up going with a lot of layers. Trains fast with large batch sizes.
	    # Integer input is the number of nodes in given layer.
	    Dense(256, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01),),
	    
	    # LeakyReLU is a modified Rectified Linear Unit.
	    # "Leaky" refers to a modification to the negative slope,
	    # which can get "stuck" in standard ReLUs.
	    LeakyReLU(alpha=0.1818),
	    Dense(32),
	    LeakyReLU(alpha=0.1818),
	    Dense(32),
	    LeakyReLU(alpha=0.1818),
	    Dense(32),
	    LeakyReLU(alpha=0.1818),
	    Dense(32),
	    LeakyReLU(alpha=0.1818),
	    Dense(32),
	    LeakyReLU(alpha=0.1818),
	    Dense(32),
	    LeakyReLU(alpha=0.1818),
	    Dense(2),
	    Activation("softmax"),
	])

	# Adam is a form of stochastic gradient descent with improved handling
	# of altering the learning rate on the fly.
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	# Compile the model with all our parameters.
	# We use binary cross-entropy, also known as log-loss.
	print("Training deep neural network model...")
	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

	return model

def main():
	# Get our data from .csv to DataFrames.
	dfs = load_data()

	# The NCAA has changed. After some tinkering I've found better results
	# excluding older data. Here we define the years for training and testing.
	train_years = [2013, 2014, 2015, 2016]
	test_years = [2009, 2010]

	# Name each DataFrame because we'll be using them a lot.
	season_results = dfs[0]
	seasons = dfs[1]
	teams = dfs[2]
	tourney_results = dfs[3]
	seeds = dfs[4]
	slots = dfs[5]

	# Calculate season stats for training and test sets.
	train_stats = calc_stats(season_results, teams, seeds, train_years, 0.1)
	test_stats = calc_stats(season_results, teams, seeds, test_years, 0.1)

	# Label and vectorize our training and test sets.
	table, labels = label(train_stats, tourney_results, train_years)
	table_test, labels_test = label(test_stats, tourney_results, test_years)

	# Massage data into correct format to be fed into DNN.
	X = normalize(table)
	y = labels
	Xt = normalize(table_test)
	yt = labels_test
	y2 = to_categorical(y)
	yt2 = to_categorical(yt)

	model = define_model()

	# Fit the model to the training set. This is the part where the machine learns.
	model.fit(X, y2, nb_epoch=1000, batch_size=256)

	# Calculate and display accuracy on the test set.
	ys_dnn = model.predict_classes(Xt, batch_size=256)
	print("Deep neural network accuracy:")
	accuracy(ys_dnn, yt)

	# Now we get to predict this year's tournament.
	test_years = [2017]
	test_stats = calc_stats(season_results, teams, seeds, test_years, 0.1)

	# This is the bracket-style prediction.
	results = predict_tournament(2017, slots, seeds, test_stats, model, teams)

	# This is the prediction of every possible matchup in the tournament.
	pred_tournaments([2017], seeds, model, test_stats)

	# Print out our final bracket.
	display_pred(results)

main()
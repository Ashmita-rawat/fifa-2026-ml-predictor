# How the model works

I'll try to explain this as clearly as I can

---

## data map

I used match results from the 2018 and 2022 world cups (128 matches total)

each match has:
- which team was home / away
- who won (or draw)
- was it a group stage game or knockout

source utlized: Fjelstul World Cup Database (CC-BY-SA 4.0) (https://github.com/jfjelstul/worldcup)


---

## step 1: elo ratings

before the model can learn anything I needed to turn team names into numbers

elo ratings does this for us. Every team starts at 1500 and after each match both teams ratings gets updated:

```
expected = 1 / (1 + 10^((opponent_elo - your_elo) / 400))
new_elo = old_elo + 32 * (actual_result - expected)
```

so if you beat a stronger team your rating goes up a lot
if you beat a weaker team it barely moves

I replayed all 128 historical matches in order so by the end each team has a rating that reflects how they actually did in 2018 and 2022

---

## step 2: logistic regression

the model takes 3 inputs per match:

1. elo gap    = (home team elo - away team elo) / 400
2. knockout   = 1 if its an elimination game, 0 if group stage
3. bias       = always 1 (this is just the intercept, so no worries here)

and return outputs which is a probability that the home team wins

formula:
```
P(home wins) = sigmoid(w1 * elo_gap + w2 * knockout + w3 * bias)

sigmoid(x) = 1 / (1 + e^(-x))
```

sigmoid squashes any number into 0-1 so we can use it as a probability

---

## step 3: training

the model training starts with random weights and gets better by:

1. making a prediction
2. comparing to the actual result
3. adjusting weights to reduce the error
4. repeating it 200 times

the loss function (how wrong the model is) is binary cross entropy:
```
loss = -mean( y * log(p) + (1-y) * log(1-p) )
```

I used mini-batch gradient descent with batch size 16 and learning rate 0.05

Then I tried 0.1 first but the weights were jumping around too much

---

## so, what did the model learned??

after training, the weights are roughly:
- w1 (elo gap): approx 2.9 - elo gap is the most important factor
- w2 (knockout): approx 1.5 - teams really do play differently in elimination games
- w3 (bias): approx -0.8 - slight correction factor

---

## step 4: monte carlo simulation (I struggled a lot here, do share your ideas, how it can be improved)
// ref: https://en.wikipedia.org/wiki/Monte_Carlo_method

once the model can predict individual matches, I then simulated the whole tournament 10,000 times

each simulation:
- plays out the group stage (round robin, probabilistic results)
- advances top 2 from each group + best 8 third place teams
- runs knockout rounds until one team wins

count how many times each team wins across 10,000 sims = championship probability

France usually comes out around 27% which matches what the bookmakers say so I think its working!!!!

---

## accuracy

about 80% on the training data

I know this sounds high but world cup matches aren't that random - a much stronger team beats a weaker team most of the time

The real limitation is this is all historical data - no injuries, no current form, no tactical stuff!!!

---

## things I want to improve

- adding more features (goals scored in qualifying, recent form)
- Trying this with a different model (random forest maybe?)  
- proper train/test split instead of just training accuracy
- account for penalty shootouts better in knockout rounds

## if you find any bugs or have suggestions feel free to open an issue!!! Would love to incorporate it!


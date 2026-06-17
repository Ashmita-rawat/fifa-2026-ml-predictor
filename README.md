# FIFA World Cup 2026 - ML Predictor

I built this because world cup is starting June 11 and I wanted to see if I could predict who wins using actual machine learning!!

live demo: https://ashmita-rawat.github.io/fifa-2026-ml-predictor/

---

## what it does

- trains a logistic regression model in the browser on real 2018 + 2022 world cup data
- uses elo ratings to figure out how strong each team is
- you pick your groups and build a bracket
- every matchup shows the ML win probability
- runs 10,000 simulations to figure out who actually has the best chance
- tells you WHY it picked your team (this part took me the longest to build)

---

## how to use

1. hit Train Model and watch the loss curve drop (this part is genuinely cool)
2. Pick Groups - click once for winner, again for runner-up
3. Bracket - click teams to advance them
4. My Prediction - see your winner + why the model picked them

---

## the ML stuff

I used logistic regression because its simple to implement

for team strength I used elo ratings (same system chess uses to rank players)
model is logistic regression 
training via mini batch gradient descent
loss function is binary cross entropy
for simulation I ran monte carlo with 10,000 runs

got about 80% accuracy on training data which I think is pretty good for football matches! Feel free to drop your opinions!

the weights the model learned:
- elo gap matters most (approx 55% of prediction)
- knockout stage matters too (approx 29%) - teams really do play differently in elimination games
- bias term is negative which basically means the model slightly favors away teams which makes sense in fifa cups

---

## data

used the Fjelstul World Cup Database (open source, CC-BY-SA 4.0)

- `wc_2018_matches.csv` - all 64 matches from russia 2018
- `wc_2022_matches.csv` - all 64 matches from qatar 2022  
- `teams_2026.csv` - all 48 teams with their starting elo ratings

---

## project structure

index.html - the running file, this is the main file, I placed everything in one place to keep it simple

wc_2018_matches.csv - match results from 2018 fifa cup
wc_2022_matches.csv - match results from 2022 fifa cup  
teams_2026.csv - all 48 teams for 2026 with their elo seeds

elo_engine.js - handles the elo rating system
logistic_regression.js - the actual ml model

methodology.md - explains how everything works in detail

---

## run it locally

```bash
git clone https://github.com/Ashmita-rawat/fifa2026-ml-predictor
cd fifa2026-ml-predictor
open index.html
```

no npm install or anything, just open the html file

---

## background

I have been getting into ML recently and thought - why not !! Will push more models!!

Took me a while to get the training loop working properly and the insight card explaining WHY the model picked a team was probably the hardest part

anyways hope its useful, open to feedback

---

## known issues / things i want to improve

- teams with no 2018/2022 history just get a default elo rating which isnt ideal
- no real time data (injuries, current form etc)
- the monte carlo simulation doesnt account for draw probability perfectly
- UI probably needs work on mobile

---

## sources

data: Fjelstul, Joshua C. (2022). The Fjelstul World Cup Database.
license: CC-BY-SA 4.0
source: https://github.com/jfjelstul/worldcup

---

## note

This model is trained on 2018 and 2022 world cup data only - it does not update with 2026 match results in real time

So, the predictions are based on historical team strength, not whats happening in the tournament right now.

That said - feel free to create your own bracket, prove the model wrong, or take live 2026 matches into consideration when making your picks. 

Upsets are literally happening as you read this!!!

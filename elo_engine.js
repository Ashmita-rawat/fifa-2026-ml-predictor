/**
 * elo_engine.js
 * 
 * Dynamic Elo rating system for football.
 * Replays historical or previous fifa matches to compute team strength ratings
 * that feed into the logistic regression model as features.
 * 
 * Based on the original Elo system (Arpad Elo, 1960) adapted for football.
 * ref: https://en.wikipedia.org/wiki/Elo_rating_system
 * FIFA uses a similar system for their official world rankings.
 * ref: https://www.fifa.com/fifa-world-ranking/procedure-men
 */

const BASE_ELO = 1500; // starting point for all teams, same as Lichess uses
// ref: https://lichess.org/faq#rating-start
// elo system was originally made for chess (Arpad Elo, 1960)
// football borrowed it because the logic is the same!!
// stronger player/team wins more often, ratings update after every match
// FIFA actually uses an elo-based system for world rankings since 2018
// higher elo = stronger team, difference between teams is what matters

const K_FACTOR = 32; // tried 64 first but ratings were going crazy
// k factor controls how much ratings change after each match
// 64 was too sensitive - one upset would swing ratings too much
// 32 is the standard, FIFA used this too before switching to their own formula

/**
 * Expected probability that team A beats team B
 * Classic Elo formula using a 400-point scale
 * ref: https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
 * 
 * @param {number} eloA - Elo rating of team A
 * @param {number} eloB - Elo rating of team B
 * @returns {number} probability between 0 and 1
 */
// well, this formula took me a while to understand!!!
function expectedScore(eloA, eloB) {
    return 1 / (1 + Math.pow(10, (eloB - eloA) / 400));
}

/**
 * Updated both teams' Elo ratings after a match result
 * 
 * @param {object} ratings - mutable ratings object {teamName: eloValue}
 * @param {string} homeTeam
 * @param {string} awayTeam  
 * @param {number} actualScore - 1 = home win, 0.5 = draw, 0 = away win
 */
// first version - just averaged the ratings
// function updateElo(ratings, home, away, score) {
//   ratings[home] = (ratings[home] + score * 100) / 2
// }
// the upper one was wrong, this is the real one:
function updateElo(ratings, homeTeam, awayTeam, actualScore) {
    const eloH = ratings[homeTeam] || BASE_ELO;
    const eloA = ratings[awayTeam] || BASE_ELO;
    const expected = expectedScore(eloH, eloA);

    ratings[homeTeam] = eloH + K_FACTOR * (actualScore - expected);
    ratings[awayTeam] = eloA + K_FACTOR * ((1 - actualScore) - (1 - expected));
}

/**
 * Replaying all the historical matches in chronological order
 * to compute final Elo ratings for each team.
 * 
 * @param {Array} matches - array of [home, away, result, knockout]
 * @param {object} seedRatings - initial Elo seeds per team
 * @returns {object} final Elo ratings after all matches
 */
function replayHistory(matches, seedRatings) {
    const ratings = { ...seedRatings };

    matches.forEach(([home, away, result]) => {
        const actualScore = result === 1 ? 1 : result === 0 ? 0.5 : 0;
        updateElo(ratings, home, away, actualScore);
    });

    return ratings;
}

/**
 * Getting the Elo rating for a team (falls back to BASE_ELO if not found)
 */
function getElo(ratings, teamName) {
    return ratings[teamName] || BASE_ELO;
}

module.exports = { expectedScore, updateElo, replayHistory, getElo, BASE_ELO, K_FACTOR };


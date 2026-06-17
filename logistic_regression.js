/**
 * logistic_regression.js
 * 
 * Binary logistic regression trained via mini-batch gradient descent.
 * Predicts the probability that the home team wins a given match.
 * 
 * Model formula:
 *   P(home wins) = sigmoid(w1 * delta_elo + w2 * knockout_stage + w3 * bias)
 * 
 * Where:
 *   delta_elo      = (home_elo - away_elo) / 400   [normalised strength gap]
 *   knockout_stage = 1 if elimination match, 0 if group stage
 *   bias           = 1 (intercept term, always)
 * 
 * Loss function: Binary cross-entropy
 *   L = -mean( y * log(p) + (1-y) * log(1-p) )
 * 
 * Optimiser: Mini-batch gradient descent
 *   w = w - learning_rate * gradient
 */

const LEARNING_RATE = 0.05; // 0.1 was too fast, 0.01 was too slow, 0.05 seems ok
const EPOCHS = 200; // tried 500 first but took forever
const BATCH_SIZE = 16; // batch size of 32 didnt work as well for some reason

/**
 * Sigmoid activation function
 * Squashes any real number into (0, 1), interpretable as a probability
 * 
 * @param {number} x
 * @returns {number} value between 0 and 1
 */
// sigmoid squashes numbers between 0 and 1 so, I did use it as probability
function sigmoid(x) {
    return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

/**
 * Building feature vector for a single match
 * 
 * @param {number} homeElo
 * @param {number} awayElo
 * @param {number} knockout - 0 or 1
 * @returns {number[]} [delta_elo, knockout, bias]
 */
function buildFeatures(homeElo, awayElo, knockout) {
    return [
        (homeElo - awayElo) / 400,  // normalised Elo gap
        knockout,                    // knockout stage flag
        1                            // bias term
    ];
}

/**
 * Forward pass: compute prediction from features and weights
 * 
 * @param {number[]} features
 * @param {number[]} weights [w1, w2, w3]
 * @returns {number} probability of home win
 */
function predict(features, weights) {
    const z = features.reduce((sum, f, i) => sum + f * weights[i], 0);
    return sigmoid(z);
}

/**
 * Computing binary cross-entropy loss and accuracy over the full dataset
 * 
 * @param {Array} dataset - array of [features, label]
 * @param {number[]} weights
 * @returns {{loss: number, accuracy: number}}
 */
function evaluate(dataset, weights) {
    let totalLoss = 0;
    let correct = 0;

    dataset.forEach(([features, label]) => {
        const p = predict(features, weights);
        const clipped = Math.max(1e-7, Math.min(1 - 1e-7, p));
        totalLoss += -(label * Math.log(clipped) + (1 - label) * Math.log(1 - clipped));
        if ((p > 0.5 && label === 1) || (p <= 0.5 && label === 0)) correct++;
    });

    return {
        loss: totalLoss / dataset.length,
        accuracy: correct / dataset.length
    };
}

/**
 * Train logistic regression via mini-batch gradient descent
 * 
 * @param {Array} dataset - array of [features, label]
 * @param {Function} onEpoch - callback(epoch, loss, accuracy) for progress
 * @returns {number[]} trained weights [w1, w2, w3]
 */
// main training function - this is where the magic happens i guess
function train(dataset, onEpoch) {
    let weights = [0.1, 0.05, 0.0];

    for (let epoch = 0; epoch < EPOCHS; epoch++) {
        // mini-batch gradient descent
        for (let b = 0; b < dataset.length; b += BATCH_SIZE) {
            const batch = dataset.slice(b, b + BATCH_SIZE);
            const gradients = [0, 0, 0];

            batch.forEach(([features, label]) => {
                const error = predict(features, weights) - label;
                features.forEach((f, i) => { gradients[i] += error * f; });
            });

            weights = weights.map((w, i) => w - LEARNING_RATE * gradients[i] / batch.length);
        }

        // reporting progress on every 10 epochs
        if (epoch % 10 === 0 && onEpoch) {
            const { loss, accuracy } = evaluate(dataset, weights);
            onEpoch(epoch, loss, accuracy);
        }
    }

    return weights;
}

/**
 * Computing win probability for a matchup using trained weights
 * 
 * @param {number} homeElo
 * @param {number} awayElo
 * @param {number} knockout
 * @param {number[]} weights
 * @returns {number} probability home team wins
 */
function winProbability(homeElo, awayElo, knockout, weights) {
    return predict(buildFeatures(homeElo, awayElo, knockout), weights);
}

// TODO: maybe try adding more features later (goals scored, recent form etc)
// TODO: add regularization?? maybe will append it later in the future!!!

module.exports = { sigmoid, buildFeatures, predict, evaluate, train, winProbability, LEARNING_RATE, EPOCHS, BATCH_SIZE };



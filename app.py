from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.stats import binom, poisson, norm, expon, pearsonr, spearmanr, kendalltau, ttest_1samp, chi2_contingency
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# --------------------- Render Pages ---------------------
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/Probability')
def Probability():
    return render_template("Probability.html")

@app.route('/Random_Variables')
def Random_Variables():
    return render_template("Random_Variables.html")

@app.route('/correlation_regression')
def correlation_regression():
    return render_template("coralation_reggresion.html")

@app.route('/Hypothesis')
def Hypothesis():
    return render_template("Hypothesis.html")

# -------------------- Probability Calculation -------------------
@app.route('/calculate_probability', methods=['POST'])
def calculate_probability():
    data = request.json
    prob_type = data['type']

    try:
        if prob_type == 'binomial':
            n = int(data['n'])
            p = float(data['p'])
            k = int(data['k'])
            if p < 0 or p > 1:
                return jsonify({'error': 'Probability p must be between 0 and 1.'}), 400
            result = binom.pmf(k, n, p)

        elif prob_type == 'poisson':
            lam = float(data['lambda'])
            k = int(data['k'])
            if lam < 0:
                return jsonify({'error': 'Lambda must be non-negative.'}), 400
            result = poisson.pmf(k, lam)

        elif prob_type == 'normal':
            mean = float(data['mean'])
            std = float(data['std'])
            x = float(data['x'])
            if std <= 0:
                return jsonify({'error': 'Standard deviation must be positive.'}), 400
            result = norm.pdf(x, mean, std)

        elif prob_type == 'exponential':
            lam = float(data['lambda'])
            x = float(data['x'])
            if lam <= 0:
                return jsonify({'error': 'Lambda must be positive.'}), 400
            result = expon.pdf(x, scale=1/lam)

        else:
            return jsonify({'error': 'Invalid probability type.'}), 400

        return jsonify({'result': round(result, 4)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/calculate_hypothesis', methods=['POST'])
def calculate_hypothesis():
    try:
        data = request.get_json()

        # Extract and validate inputs
        test_type = data.get('test_type')
        sample_data = data.get('sample_data')
        population_mean = float(data.get('population_mean', 0))

        if not sample_data or not isinstance(sample_data, list):
            return jsonify({'error': 'Invalid sample data format.'}), 400

        sample_data = [float(x) for x in sample_data]

        if not sample_data or len(sample_data) == 0:
            return jsonify({'error': 'No sample data provided.'}), 400

        # Z-Test
        if test_type == 'z-test':
            mean = np.mean(sample_data)
            std = np.std(sample_data, ddof=1)

            if std == 0:
                return jsonify({'error': 'Standard deviation cannot be zero.'}), 400

            z_score = (mean - population_mean) / (std / np.sqrt(len(sample_data)))
            p_value = 2 * (1 - norm.cdf(abs(z_score)))

            return jsonify({
                'test_statistic': round(z_score, 4),
                'p_value': round(p_value, 4)
            })

        # T-Test
        elif test_type == 't-test':
            t_statistic, p_value = ttest_1samp(sample_data, population_mean)

            return jsonify({
                'test_statistic': round(t_statistic, 4),
                'p_value': round(p_value, 4)
            })

        # Chi-Square Test
        elif test_type == 'chi-square':
            if len(sample_data) != 4:
                return jsonify({'error': 'Chi-square test requires exactly 4 values.'}), 400

            # Reshape into 2x2 matrix for Chi-Square
            observed = np.array(sample_data).reshape(2, 2)
            
            # Disable Yates' correction to get accurate Chi-Square statistic for 2x2 tables
            chi2, p_value, _, _ = chi2_contingency(observed, correction=False)

            return jsonify({
                'test_statistic': round(chi2, 4),
                'p_value': round(p_value, 4)
            })

        else:
            return jsonify({'error': 'Invalid test type.'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Correlation and Regression -------------------
@app.route('/calculate_correlation', methods=['POST'])
def calculate_correlation():
    data = request.json

    try:
        x_values = list(map(float, data['x_values']))
        y_values = list(map(float, data['y_values']))
        calculation_type = data['calculation_type']

        # Input validation
        if len(x_values) != len(y_values):
            return jsonify({'error': 'X and Y values must have the same length.'}), 400

        # Perform the selected correlation or regression
        if calculation_type == 'pearson':
            result, _ = pearsonr(x_values, y_values)
        elif calculation_type == 'spearman':
            result, _ = spearmanr(x_values, y_values)
        elif calculation_type == 'kendall':
            result, _ = kendalltau(x_values, y_values)
        elif calculation_type == 'linear_regression':
            model = LinearRegression().fit(np.array(x_values).reshape(-1, 1), y_values)
            result = {
                'Intercept': round(model.intercept_, 4),
                'Slope': round(model.coef_[0], 4)
            }
        else:
            return jsonify({'error': 'Invalid calculation type.'}), 400

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# -------------------- Flask App Runner -------------------
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT',10000))
    app.run(host='0.0.0.0',port=port)
    app.run(debug=True)

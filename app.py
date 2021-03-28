from flask import Flask, render_template, redirect, url_for, request, session, flash
from functools import wraps
import pickle
import numpy as np

app = Flask(__name__)

# config for session data
app.secret_key = 'elvino2021'


# login required decorator
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))

    return wrap


@app.route('/')
@login_required
def home():
    return render_template('main.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            session['logged_in'] = True
            flash('You were logged in.')
            return redirect(url_for('home'))
    return render_template('login.html', error=error)


@app.route('/logout')
@login_required
def logout():
    session.pop('logged_in', None)
    flash('You were logged out.')
    return redirect(url_for('login'))


wineML = pickle.load(open('../wine_model.pkl', 'rb'))


@app.route('/results', methods=['GET', 'POST'])
def results():
    feature_values = [float(x) for x in request.form.values()]
    final = [np.array(feature_values, dtype=float)]
    print(feature_values)
    print(final)
    prediction = wineML.predict(final)
    print(prediction)
    if prediction == 0:
        return render_template('results.html', result='Wine Quality: Okay Wine')
    else:
        return render_template('results.html', result='Wine Quality: Great Wine')
    # return render_template('results.html', result=feature_values)


if __name__ == "__main__":
    app.run(debug=True)

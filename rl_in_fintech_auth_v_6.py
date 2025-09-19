# rl_in_fintech_auth_v6.py
# Fix: include user_age in form and inference to avoid "age group" inference errors.
# Streamlit RL FinTech app v6

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import requests
import hashlib
import secrets
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ---------------------------
# CONFIG / FILENAMES
# ---------------------------
USERS_FILE = "users_v6.csv"
LOG_FILE = "transactions_log_v6.csv"
MODEL_FILE = "ppo_fraud_v6"
SCALER_FILE = "scaler_v6.pkl"
FEATURES_FILE = "feature_columns_v6.json"
TRAIN_TIMESTEPS = 20000  # shorter for demo; increase for production

# ---------------------------
# CURRENCIES (demo fixed rates)
# ---------------------------
CURRENCIES = {
    "USA": ("USD", 1.0),
    "India": ("INR", 0.012),
    "Germany": ("EUR", 1.07),
    "UK": ("GBP", 1.25),
    "Iran": ("IRR", 0.000024),
    "North Korea": ("KPW", 0.0011)
}

# ---------------------------
# Simple i18n strings (English only for brevity)
# ---------------------------
S = {
    'title': 'RL FinTech Transaction Analyzer',
    'login': 'Login', 'register': 'Register', 'username': 'Username', 'password': 'Password',
    'create_account': 'Create Account', 'logout': 'Log out', 'submit_txn': 'Submit Transaction',
    'amount': 'Amount', 'currency': 'Currency', 'txn_type': 'Transaction Type', 'merchant_cat': 'Merchant Category',
    'international_country': 'Destination Country', 'time_of_day': 'Time of Day', 'user_age': 'User Age',
    'location_trust': 'Location Trust Score', 'device_trust': 'Device Trust Score',
    'txns_last_hour': 'Transactions in last hour', 'avg_amount_7d': 'Avg amount last 7 days',
    'result_fraud': '🚨 Transaction Flagged as FRAUD!', 'result_valid': '✅ Transaction Accepted as VALID.',
    'reason': 'Reason', 'recent_txns': 'Recent Transactions', 'training_info': 'Model training may run on first start and take time.'
}

# ---------------------------
# PASSWORD HASH (fallback salted sha256)
# ---------------------------
try:
    import bcrypt
    def hash_pw(pw: str) -> str:
        return bcrypt.hashpw(pw.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    def verify_pw(pw: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(pw.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False
except Exception:
    def hash_pw(pw: str) -> str:
        salt = secrets.token_hex(16)
        digest = hashlib.sha256((salt + pw).encode('utf-8')).hexdigest()
        return f"{salt}${digest}"
    def verify_pw(pw: str, hashed: str) -> bool:
        try:
            salt, dig = hashed.split('$',1)
            return hashlib.sha256((salt + pw).encode('utf-8')).hexdigest() == dig
        except Exception:
            return False

# ---------------------------
# USER DB helpers
# ---------------------------
def ensure_users_file():
    if not os.path.exists(USERS_FILE):
        pd.DataFrame(columns=['username','password']).to_csv(USERS_FILE, index=False)

def load_users():
    ensure_users_file()
    df = pd.read_csv(USERS_FILE)
    return dict(zip(df['username'].astype(str), df['password'].astype(str)))

def register_user(username: str, password: str):
    users = load_users()
    if username in users:
        return False, 'Username exists'
    hashed = hash_pw(password)
    pd.DataFrame([[username, hashed]], columns=['username','password']).to_csv(USERS_FILE, mode='a', header=not os.path.exists(USERS_FILE) or os.path.getsize(USERS_FILE)==0, index=False)
    return True, 'Registered'

# ---------------------------
# Synthetic data + preprocess
# ---------------------------
def generate_data_and_prep(n_samples=3000):
    np.random.seed(42)
    data = pd.DataFrame({
        'amount': np.random.normal(120, 80, n_samples).clip(1),
        'time': np.random.randint(0,24,n_samples),
        'user_age': np.random.randint(18,75,n_samples),
        'txns_last_hour': np.random.poisson(2, n_samples),
        'avg_amount_7d': np.random.normal(85,60,n_samples).clip(1),
        'txn_type': np.random.choice(['crypto','merchant','international'], n_samples),
        'merchant_cat': np.random.choice(['electronics','fashion','groceries','travel'], n_samples),
        'country': np.random.choice(['USA','India','Germany','Iran','North Korea','UK'], n_samples)
    })

    def label(r):
        if r['country'] in ['North Korea','Iran']:
            return 1
        if r['txn_type']=='crypto' and r['amount']>350:
            return 1
        if r['txn_type']=='international' and r['amount']>700:
            return 1
        if r['merchant_cat']=='electronics' and r['amount']>1500:
            return 1
        if r['txns_last_hour']>=12:
            return 1
        return 0

    data['label'] = data.apply(label, axis=1)
    cat = pd.get_dummies(data[['txn_type','merchant_cat','country']], prefix_sep='=')
    features = data[['amount','time','user_age','txns_last_hour','avg_amount_7d']].join(cat)
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    scaled['label'] = data['label'].values
    return scaled, scaler, list(features.columns)

# ---------------------------
# Gym environment
# ---------------------------
class FraudEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(df.shape[1]-1,), dtype=np.float32)
        self.i = 0
    def reset(self, seed=None, options=None):
        self.i = 0
        return self.df.iloc[self.i,:-1].values.astype(np.float32), {}
    def step(self, action):
        label = int(self.df.iloc[self.i]['label'])
        reward = 1 if int(action)==label else -1
        self.i += 1
        done = self.i>=len(self.df)
        obs = self.df.iloc[self.i,:-1].values.astype(np.float32) if not done else np.zeros(self.df.shape[1]-1, dtype=np.float32)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, {}

# ---------------------------
# Hard rules
# ---------------------------
def apply_hard_rules(txn: dict):
    if txn['country'] in ['North Korea','Iran']:
        return True, 'Sanctioned/high-risk country'
    if txn['txn_type']=='crypto' and txn['amount_usd']>500:
        return True, 'Large crypto amount'
    if txn['txn_type']=='international' and txn['amount_usd']>1000:
        return True, 'Large international transfer'
    if txn.get('merchant_cat','')=='electronics' and txn['amount_usd']>2000:
        return True, 'High-value electronics'
    if txn.get('txns_last_hour',0)>=15:
        return True, 'High transaction velocity'
    if txn.get('location_trust',1.0)<0.2 and txn['amount_usd']>200:
        return True, 'Low location trust with high amount'
    return False, ''

# ---------------------------
# Prepare features for inference
# ---------------------------
def prepare_features(input_txn: dict, scaler: StandardScaler, feature_cols: list):
    numeric = pd.DataFrame([{
        'amount': input_txn['amount_usd'],
        'time': input_txn['time'],
        'user_age': input_txn['user_age'],
        'txns_last_hour': input_txn['txns_last_hour'],
        'avg_amount_7d': input_txn['avg_amount_7d']
    }])
    cat = pd.get_dummies(pd.DataFrame([{
        'txn_type': input_txn['txn_type'],
        'merchant_cat': input_txn.get('merchant_cat',''),
        'country': input_txn['country']
    }]), prefix_sep='=')
    merged = pd.concat([numeric, cat], axis=1)
    for c in feature_cols:
        if c not in merged.columns:
            merged[c] = 0
    merged = merged.reindex(columns=feature_cols, fill_value=0)
    scaled = pd.DataFrame(scaler.transform(merged), columns=feature_cols)
    return scaled

# ---------------------------
# Load or train model
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_model_and_preprocessors(train_if_missing=True, timesteps=TRAIN_TIMESTEPS):
    need_train = not (os.path.exists(MODEL_FILE + '.zip') and os.path.exists(SCALER_FILE) and os.path.exists(FEATURES_FILE))
    if need_train and train_if_missing:
        st.info(S['training_info'])
        df, scaler, feature_cols = generate_data_and_prep(n_samples=4000)
        env = FraudEnv(df)
        model = PPO('MlpPolicy', env, verbose=0)
        model.learn(total_timesteps=timesteps)
        model.save(MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        with open(FEATURES_FILE,'w') as f:
            json.dump(feature_cols,f)
        return model, scaler, feature_cols
    else:
        model = PPO.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        with open(FEATURES_FILE,'r') as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols

# ---------------------------
# Detect country by IP (best-effort)
# ---------------------------
def detect_country_ip():
    try:
        r = requests.get('https://ipinfo.io/json', timeout=2)
        if r.status_code==200:
            js = r.json()
            iso = js.get('country')
            iso_map = {'US':'USA','IN':'India','DE':'Germany','IR':'Iran','KP':'North Korea','GB':'UK'}
            return iso_map.get(iso, iso) if iso else 'Unknown'
    except Exception:
        pass
    return 'Unknown'

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title='RL FinTech v6', layout='centered')

# Ensure users file
ensure_users_file()

# Auth state
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

st.title(S['title'])
col1, col2 = st.columns(2)
with col1:
    st.subheader(S['login'])
    login_user = st.text_input(S['username'], key='login_user')
    login_pass = st.text_input(S['password'], type='password', key='login_pass')
    if st.button(S['login'] + ' 🔑'):
        users = load_users()
        stored = users.get(str(login_user))
        if stored and verify_pw(login_pass, stored):
            st.session_state['authenticated'] = True
            st.session_state['user'] = login_user
            st.success(f"{login_user} — logged in")
            st.rerun()
        else:
            st.error('❌ Login failed')
with col2:
    st.subheader(S['register'])
    reg_user = st.text_input('New ' + S['username'], key='reg_user')
    reg_pass = st.text_input('New ' + S['password'], type='password', key='reg_pass')
    reg_pass2 = st.text_input('Re-enter ' + S['password'], type='password', key='reg_pass2')
    if st.button(S['create_account'] + ' ➕'):
        if not reg_user or not reg_pass:
            st.error('Enter username and password')
        elif reg_pass != reg_pass2:
            st.error('Passwords do not match')
        else:
            ok, msg = register_user(reg_user, reg_pass)
            if ok:
                st.success('✅ ' + msg + '. Please login.')
            else:
                st.error('❌ ' + msg)

if not st.session_state.get('authenticated', False):
    st.info('Please login or register')
    st.stop()

st.sidebar.success(f"{st.session_state['user']}")
if st.sidebar.button(S['logout'] + ' ⎋'):
    st.session_state['authenticated'] = False
    st.session_state['user'] = None
    st.rerun()

# Load model
with st.spinner('Loading model and preprocessors (may train on first run)...'):
    model, scaler, feature_cols = get_model_and_preprocessors(train_if_missing=True)

# Country & currency
detected = detect_country_ip()
all_countries = list(CURRENCIES.keys())
if detected not in all_countries:
    detected = 'USA'
country = st.selectbox(S['currency'] + ' / Country', options=all_countries, index=all_countries.index(detected), help='Auto-detected; override if needed')
currency_symbol, rate_to_usd = CURRENCIES[country]

# Transaction form (now includes user_age)
st.subheader(S['submit_txn'])
with st.form('txn'):
    local_amount = st.number_input(f"{S['amount']} ({currency_symbol})", min_value=0.01, value=100.0, format="%.2f")
    amount_usd = float(local_amount) * float(rate_to_usd)
    st.write(f"Converted amount: ${amount_usd:.2f} USD")

    txn_type = st.selectbox(S['txn_type'], options=['merchant','crypto','international'])
    merchant_cat = ''
    if txn_type == 'merchant':
        merchant_cat = st.selectbox(S['merchant_cat'], options=['electronics','fashion','groceries','travel'])
    international_country = ''
    if txn_type == 'international':
        international_country = st.selectbox(S['international_country'], options=all_countries)

    time_of_day = st.slider(S['time_of_day'], 0, 23, datetime.now().hour)
    user_age = st.slider(S['user_age'], 18, 75, 30)
    location_trust = st.slider(S['location_trust'], 0.0, 1.0, 0.8)
    device_trust = st.slider(S['device_trust'], 0.0, 1.0, 0.9)
    txns_last_hour = st.number_input(S['txns_last_hour'], min_value=0, value=0)
    avg_amount_7d = st.number_input(S['avg_amount_7d'], min_value=0.0, value=80.0, format='%.2f')

    submit = st.form_submit_button(S['submit_txn'])

if submit:
    txn = {
        'local_amount': local_amount,
        'amount_usd': amount_usd,
        'currency': currency_symbol,
        'country': country if txn_type!='international' else international_country,
        'txn_type': txn_type,
        'merchant_cat': merchant_cat,
        'time': int(time_of_day),
        'user_age': int(user_age),
        'location_trust': float(location_trust),
        'device_trust': float(device_trust),
        'txns_last_hour': int(txns_last_hour),
        'avg_amount_7d': float(avg_amount_7d),
        'user': st.session_state['user']
    }

    # 1) Hard rule checks
    is_fraud, reason = apply_hard_rules(txn)
    if is_fraud:
        final = 'Fraud'
        st.error(S['result_fraud'])
        st.write(S['reason'] + ': ' + reason)
    else:
        # 2) RL inference
        try:
            X = prepare_features(txn, scaler, feature_cols)
            obs = X.values.astype('float32')
            action, _ = model.predict(obs, deterministic=True)
            final = 'Fraud' if int(action[0])==1 else 'Valid'
            if final == 'Fraud':
                st.error(S['result_fraud'])
            else:
                st.success(S['result_valid'])
            reason = 'RL model decision'
        except Exception as e:
            final = 'Fraud'
            reason = f'Inference error: {e}'
            st.error('Inference error — flagged for manual review')

    # Save transaction (include user_age)
    row = {
        'Timestamp': datetime.now(), 'User': txn['user'], 'local_amount': txn['local_amount'], 'currency': txn['currency'],
        'amount_usd': txn['amount_usd'], 'country': txn['country'], 'txn_type': txn['txn_type'], 'merchant_cat': txn['merchant_cat'],
        'time': txn['time'], 'user_age': txn['user_age'], 'location_trust': txn['location_trust'], 'device_trust': txn['device_trust'],
        'txns_last_hour': txn['txns_last_hour'], 'avg_amount_7d': txn['avg_amount_7d'], 'Prediction': final, 'Reason': reason
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(LOG_FILE):
        df_row.to_csv(LOG_FILE, index=False, header=False, mode='a')
    else:
        df_row.to_csv(LOG_FILE, index=False)

    st.balloons()

# Recent transactions
st.markdown('---')
st.subheader(S['recent_txns'])
if os.path.exists(LOG_FILE):
    logs = pd.read_csv(LOG_FILE)
    st.dataframe(logs.tail(10))
else:
    st.info('No transactions yet')

st.caption('Made with rules + RL (PPO).')

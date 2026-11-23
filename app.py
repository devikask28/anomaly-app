import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Transaction Monitor', layout='wide')

# ---- load the trained model (made in the Colab notebook) ----
bundle = joblib.load('model.joblib')
model = bundle['model']
history_scores = bundle['history_scores']
features = bundle['features']

# Required fields for the incoming transactions
REQUIRED_COLUMNS = [
    "transaction_id",
    "amount",
    "customer_typical_amount",
    "transactions_last_hour",
    "minutes_since_previous",
    "account_age_days"
]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]

def validate_transaction(record):
    transaction_id = record.get("transaction_id")
    if pd.isna(transaction_id) or not str(transaction_id).strip():
        return "missing transaction ID"
    values = {}
    for column in NUMERIC_COLUMNS:
        value = pd.to_numeric(record.get(column), errors="coerce")
        if pd.isna(value) or not np.isfinite(value):
            return f"missing or non-numeric value in '{column}'"
        values[column] = float(value)
    if values["amount"] <= 0:
        return "amount must be greater than zero"
    if values["customer_typical_amount"] <= 0:
        return "customer's typical amount must be greater than zero"
    for column in [
        "transactions_last_hour",
        "minutes_since_previous",
        "account_age_days",
    ]:
        if values[column] < 0:
            return f"'{column}' cannot be negative"
    if not values["transactions_last_hour"].is_integer():
        return "'transactions_last_hour' must be a whole number"
    return None

def score(amount, typical, last_hour, gap, age):
    """Turn one transaction into the five features and return its anomaly score."""
    row = pd.DataFrame([{
        'log_amount': np.log1p(amount),
        'amount_ratio': amount / typical,
        'transactions_last_hour': last_hour,
        'log_gap': np.log1p(gap),
        'log_age': np.log1p(age),
    }])[features]
    return -model.score_samples(row)[0]


# ---- sidebar: the review budget ----
st.title('Transaction Monitor')
st.write('Each new transaction is scored by Isolation Forest. '
         'If the score is above the cutoff, it is recommended for human review.')

pct = st.sidebar.slider('Send this share of transactions to review (%)', 1, 10, 4)
cutoff = np.percentile(history_scores, 100 - pct)
st.sidebar.write(f'Score cutoff: **{cutoff:.3f}**')
st.sidebar.caption('The cutoff is set from past transactions: '
                   'a transaction scoring above it is more unusual than the top '
                   f'{pct}% of history.')

tab1, tab2 = st.tabs(['Check one transaction', 'Live stream'])

# ---- tab 1: type in one transaction ----
with tab1:
    c1, c2 = st.columns(2)
    amount = c1.number_input('Amount ($)', min_value=1.0, value=60.0)
    typical = c1.number_input("Customer's typical amount ($)", min_value=1.0, value=50.0)
    last_hour = c1.number_input('Transactions in the last hour', min_value=0, value=1)
    gap = c2.number_input('Minutes since previous transaction', min_value=0.0, value=120.0)
    age = c2.number_input('Account age (days)', min_value=0, value=700)

    if st.button('Score this transaction'):
        s = score(amount, typical, last_hour, gap, age)
        if s > cutoff:
            st.error(
                f'**UNUSUAL - HUMAN REVIEW RECOMMENDED** (score {s:.3f}, cutoff {cutoff:.3f})\n\n'
                'This transaction is unusual compared with past transactions. That does not '
                'mean it is fraud. **Next step:** investigate it, for example by checking the '
                'account history or contacting the customer, before taking any action.'
            )
        else:
            st.success(
                f'**NO UNUSUAL PATTERN DETECTED** (score {s:.3f}, cutoff {cutoff:.3f})\n\n'
                'This transaction looks ordinary compared with past transactions. That does '
                'not mean it is legitimate. **Next step:** no review is required by the model, '
                'but review it if you have other reasons for concern.'
            )
        share = (history_scores < s).mean() * 100
        st.write(f'This is more unusual than {share:.0f}% of past transactions. This is a rank, not a probability of fraud.')

# ---- tab 2: transactions arriving one by one ----
with tab2:
    st.write('Press the button. The app receives the new transactions one at a time and scores each one as it arrives.')
    speed = st.select_slider('Speed', options=['Slow', 'Medium', 'Fast'], value='Medium')
    pause = {'Slow': 0.5, 'Medium': 0.15, 'Fast': 0.02}[speed]

    if st.button('Start the stream'):
        try:
            stream = pd.read_csv('new_transactions.csv')
            missing = [c for c in REQUIRED_COLUMNS if c not in stream.columns]
            if missing:
                raise ValueError('missing columns: ' + ', '.join(missing))
            if stream.empty:
                raise ValueError('the file has no transactions')
        except Exception as exc:
            st.error(f'Cannot start the stream: `new transactions.csv` could not be used ({exc}). No transactions were scored. Upload a complete file and try again.')
            st.stop()
        stream[NUMERIC_COLUMNS] = stream[NUMERIC_COLUMNS].apply(pd.to_numeric, errors='coerce')
        box = st.empty()
        counts = st.empty()
        st.subheader('Recommended for human review')
        table = st.empty()
        flagged, skipped = [], []

        for i, r in stream.iterrows():
            reason = validate_transaction(r)
            if reason:
                skipped.append({'transaction_id': r['transaction_id'], 'reason': reason})
                box.warning(f"{r['transaction_id']}: NOT SCORED ({reason})")
                continue
            s = score(r['amount'], r['customer_typical_amount'],
                      r['transactions_last_hour'], r['minutes_since_previous'],
                      r['account_age_days'])
            msg = f"{r['transaction_id']}: ${r['amount']:.2f}  score {s:.3f}"
            if s > cutoff:
                flagged.append({'id': r['transaction_id'], 'amount': r['amount'],
                                'typical': r['customer_typical_amount'],
                                'last_hour': r['transactions_last_hour'],
                                'score': round(s, 3)})
                box.error(msg + '  ->  UNUSUAL, review recommended (not confirmed fraud)')
            else:
                box.success(msg + '  ->  no unusual pattern (not an approval)')
            counts.write(f'Received **{i + 1}** of {len(stream)}.  '
                         f'Sent to review: **{len(flagged)}**')
            if flagged:
                table.dataframe(pd.DataFrame(flagged), hide_index=True)
            time.sleep(pause)

        if skipped:
            st.warning(f'Batch not complete: {len(skipped)} of {len(stream)} transactions were NOT scored. They are not "OK". Correct them and resubmit, or review them manually.')
            st.dataframe(pd.DataFrame(skipped), hide_index=True)
        else:
            st.success(f'Done. {len(flagged)} transactions should be sent for human review.')

        # if the file has an answer key, check how we did
        if 'needs_review' in stream.columns:
            answer = stream.set_index('transaction_id')['needs_review']
            hits = sum(answer[f['id']] for f in flagged)
            st.write(f'{hits} of the {len(flagged)} flagged transactions really needed review. In total, {answer.sum()} transactions needed review.')

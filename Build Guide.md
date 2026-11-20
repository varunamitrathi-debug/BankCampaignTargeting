# Build guide: bank campaign targeting in Python

One or two evenings.

## 1. Get the data

Download [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing). The zip contains two more zips; open the bank additional one and take the full file with 41,188 rows. Save it in `data/` as `bank marketing.csv`. It uses semicolons, which is why the script reads it with `sep=";"`.

## 2. Run and read along

```
pip install pandas scikit-learn matplotlib
python analysis.py
```

The script has five steps:

* **Step 1 Explore.** `groupby` gives the subscription rate for each value of a column. Start here in any targeting project: simple rates often tell most of the story (previous outcome, channel, number of calls).
* **Step 2 Model.** The pipeline turns text columns into 0 or 1 columns (one hot encoding), puts numeric columns on the same scale, then fits a logistic regression. Duration of the call is excluded because it is only known after the call.
* **Step 3 Deciles.** Score the 30% test set the model never saw, sort from highest to lowest, and cut into ten equal groups. Lift = a group's rate divided by the average rate. AUC (0.803 here) measures how well the scores rank buyers above non buyers; 0.5 is a coin toss.
* **Step 4 Sizing.** Two assumptions sit at the top of this step:

```
costPerCall = 5.0
valuePerSubscriber = 100.0
```

Net value = subscribers x value minus calls x cost, for calling the top 3, 4, 5 or all 10 deciles. Change the two numbers and rerun to see where the best cut off moves.

* **Step 5 Chart.** The cumulative gains curve: how many subscribers you reach for each share of customers called. The further it bows above the diagonal, the better the ranking.

## 3. Write up

Lead with the decision (whom to call), show the decile table, name the two assumptions, and explain why duration was left out; interviewers often ask.

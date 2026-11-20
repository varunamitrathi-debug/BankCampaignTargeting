# Bank campaign targeting: who should we call?

**Question.** A Portuguese bank phoned 41,188 customers to sell a term deposit and 11.3% said yes. Which customers should it call next time, and what is better targeting worth?

**Answer in one line.** A simple, explainable model ranks customers well (AUC 0.803 on unseen data): the top 10% of scores convert at 50.7%, and calling only the top half reaches 83% of subscribers with half the calls, which under the cost assumptions below raises net campaign value by 9.4%.

## Approach

* **Data.** UCI Bank Marketing (bank additional full): 41,188 calls from May 2008 to November 2010, with customer profile, campaign history and the economic indicators at the time of the call.
* **Explore.** Subscription rate by previous campaign outcome, channel, month, job and number of calls.
* **Model.** Logistic regression in Python (scikit learn) on a 70/30 split. Call duration is left out on purpose: it is only known after the call, so using it would make the model look better than it can be in practice.
* **Size it.** Rank the 12,357 test customers by score, cut them into ten equal groups, and compare calling plans under stated cost assumptions.

## Key findings

* **Past success is the strongest signal.** Customers who said yes in a previous campaign subscribed 65.1% of the time, against 14.2% after a previous no and 8.8% for customers never contacted before.
* **Mobile beats landline.** 14.7% on mobile against 5.2% on a landline.
* **More calls, fewer yeses.** 13.0% subscribe on the first call of a campaign; by the tenth call or later it is 3.6%.
* **The ranking works.** On the hold out set, the top 10% of scores converted at 50.7% (4.5 times the average), and the top 30% contained 72.6% of all subscribers.

| Plan (test set) | Calls | Subscribers | Share of subscribers | Net value, EUR (assumption based) |
|---|---|---|---|---|
| Call top 30% | 3,708 | 1,010 | 72.6% | 82,460 |
| Call top 50% | 6,179 | 1,156 | 83.0% | 84,705 |
| Call everyone | 12,357 | 1,392 | 100% | 77,415 |

![Cumulative gains](charts/cumulative%20gains.png)

## Recommendation

* **Call the top half of the model's scores first.** Under the assumed €5 cost per call and €100 margin per deposit, this earns €84,705 on the test set against €77,415 for calling everyone (9.4% more) with 50% fewer calls.
* **Cap attempts at five calls per customer.** From the fifth call onward, conversion is below 8%.
* **Prioritise previous subscribers and mobile numbers** in the call list; they are the two clearest signals in the data.

Cost per call and value per subscriber are **assumptions** kept at the top of step 4 in `analysis.py`; the best cut off moves if they change, which is exactly what the table is for.

## Files

```
BankCampaignTargeting/
  README.md           this summary
  Build Guide.md      step by step instructions
  analysis.py         exploration, model, deciles, sizing and chart
  output/             tables and summary.txt with every number above
  charts/             cumulative gains chart
  data/               put the UCI CSV here (not committed)
```

## Tools

Python (pandas, scikit learn logistic regression with one hot encoding, matplotlib), lift and gains analysis.

## How to run

1. Download [Bank Marketing from UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing), unzip it, then unzip the bank additional zip inside it. Save the full CSV (41,188 rows, semicolon separated) in `data/` as `bank marketing.csv`.
2. Run:

```
pip install pandas scikit-learn matplotlib
python analysis.py
```

Data: Moro, S., Rita, P., and Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. CC BY 4.0.

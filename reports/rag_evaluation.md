# Document Q&A: evaluation

Test database: 54 SARS guides, 8,317 chunks, ingested through the same pipeline as the web app. Embeddings: bge-small-en-v1.5; re-ranker: ms-marco-MiniLM-L-6-v2; answers: Llama-3.2-3B-Instruct-Q4_K_M.gguf (llama.cpp, CPU).

## Retrieval

For each of the 30 answerable questions, is a passage that contains the answer among the 5 passages given to the model?

| | Correct passage in top 5 | Mean reciprocal rank |
|---|---|---|
| HNSW only | 28/30 (93%) | 0.732 |
| HNSW top 20, re-ranked to 5 | 28/30 (93%) | 0.759 |

## Answers

| Measure | Result |
|---|---|
| Answerable questions answered correctly (every key fact present) | 25/30 (83%) |
| Answerable questions where the model said it didn't know | 0/30 (0%) |
| Answers citing a passage that contains the answer | 21/30 (70%) |
| Answers whose numbers all appear in the cited passages | 28/30 (93%) |
| Unanswerable questions where the model said it didn't know | 6/6 (100%) |

## Speed (per question, CPU)

- Search: 24 ms
- Re-ranking 20 passages: 291 ms
- Writing the answer: 15.5 s

## Ingestion and OCR

- 54 documents, 2,671 pages ingested in 6.2 min (0.14 s per page); 5 pages needed OCR.
- OCR check: 9 pages that also have a text layer were OCR'd at 300 dpi; on average 85.7% of the real words were recovered in order, at 2.9 s per page.

## Every question

| # | Question | Correct | Rank (HNSW → re-ranked) | Answer |
|---|---|---|---|---|
| 1 | What age group must young employees fall in for an employer to claim the Employment Tax Incentive? | yes | 2 → 1 | The age group for young employees to claim the Employment Tax Incentive is between 18 and 29 years old at the end of the month in which the incentive is claimed [1]. |
| 2 | What is the rate of dividends tax? | yes | 1 → 1 | The rate of dividends tax is 20% [2, 4, 5]. |
| 3 | Are gains on shares held as trading stock for a long time taxed as capital or as income? | no | 2 → 3 | Gains on shares held as trading stock are taxed as income, not capital. [1] |
| 4 | When does a foreign supplier of electronic services have to register for VAT in South Africa? | yes | 1 → 1 | A foreign electronic services supplier must register for VAT in South Africa if their taxable supplies made to persons in South Africa exceed the R1 million registration threshold, or if they voluntarily registered for VAT, or if they were required to account for VAT on supplies made by the foreign electronic services supplier before 1 April 2025, or if they have a written agreement with the intermediary to treat supplies made on behalf of the foreign electronic services supplier as their own, effective from 1 April 2025 [1, 2, 4]. |
| 5 | What turnover limit must a business stay under to qualify for turnover tax as a micro business? | yes | 1 → 1 | According to [1] and [5], a micro business must have a qualifying turnover of R1 million or less to qualify for turnover tax. |
| 6 | How long does a taxpayer have to present a case under the mutual agreement procedure of a tax treaty? | yes | 1 → 1 | The case must be presented within three years from the first notification of the action resulting in taxation not in accordance with the provisions of the Convention [1]. |
| 7 | How much is the monthly medical scheme fees tax credit for the taxpayer? | no | – → – | The monthly medical scheme fees tax credit for the taxpayer is R286 [1]. |
| 8 | What estate duty rate applies to estates worth more than R30 million? | yes | 1 → 3 | The estate duty rate that applies to estates worth more than R30 million is 25% [3], [4], and [5]. |
| 9 | How many days does a taxpayer have to appeal after SARS disallows an objection? | yes | 1 → 1 | A taxpayer has 30 business days after delivery of the notice of disallowance or partial disallowance to appeal against the decision. |
| 10 | What makes a testamentary trust for relatives a type-B special trust? | yes | 2 → 1 | A testamentary trust for relatives is a type-B special trust if the youngest beneficiary is under the age of 18 years on the last day of the year of assessment [4.4.3]. |
| 11 | How is the notional input tax that a motor dealer may deduct on second-hand vehicles limited? | yes | 1 → 1 | The notional input tax that a motor dealer may deduct on second-hand vehicles is limited to the tax fraction of the lesser of the consideration in money given by the dealer or the OMV of the vehicle [1], [2], [4]. |
| 12 | What counts as an insolvent estate for income tax purposes? | yes | 2 → 2 | An insolvent estate for income tax purposes is defined as an estate under sequestration, as per section 1(1) of the Act, read with section 2 of the Insolvency Act [4]. |
| 13 | When is the first provisional tax payment due? | yes | 3 → 5 | The first provisional tax payment is due on 29 August 2025 [2]. |
| 14 | Since when must a buyer withhold tax when a non-resident sells property in South Africa? | yes | 1 → 1 | The withholding tax on amounts paid by the Purchaser to Non-Resident sellers of Immovable Property situated within South Africa, as prescribed in Section 35A of the Income Tax Act, came into effect from 1 September 2007 [1]. |
| 15 | Under which section of the Income Tax Act are public benefit organisations approved? | yes | 3 → 3 | Public benefit organisations are approved under section 30 of the Income Tax Act. [1] |
| 16 | By what date must VAT returns be submitted when filing manually compared to eFiling? | yes | 5 → 1 | For manual submission, VAT returns must be submitted by the 25th of the month following the end of the vendor's tax period. [2] [4] For eFiling, VAT returns must be submitted by the last business day of the month following the end of the vendor's tax period. [2] |
| 17 | What is an auto-assessment of an individual's income tax return? | yes | 1 → 1 | An auto-assessment of an individual's income tax return is an original assessment based on estimate by SARS, using information readily available to SARS, as referred to as an "automatic assessment" or "auto-assessment" [2, 3]. This type of assessment is based on Third Party data available at SARS' disposal, including data from employers, financial institutions, and other third-party data providers [4]. SARS will issue a "Notice of Assessment (ITA34)" to auto-assessed taxpayers, which will indicate "original estimate assessment" [4]. |
| 18 | How do I ask SARS to reduce an existing personal income tax assessment? | yes | 1 → 1 | You can submit a Request for Reduced Assessment (RRA01) form via eFiling to request SARS to reduce an existing assessment. This form allows taxpayers a less formal mechanism to request SARS to amend an assessment without having to follow the objection and appeal route. [1] |
| 19 | How soon must a new employer register for employees' tax, SDL and UIF? | yes | 3 → 1 | Every Employer who is obliged to register for Employees’ Tax, SDL and / or UIF purposes must apply within 21 days after becoming an Employer to the Commissioner in the prescribed form (e.g. EMP101 or EMP102) [1]. |
| 20 | How much transfer duty is payable on the first R1 100 000 of a property's value? | yes | 1 → 1 | The transfer duty payable on the first R1 100 000 of a property's value is R0 [1]. |
| 21 | Until when is the urban development zone allowance available for buildings? | no | – → – | The urban development zone allowance is available for buildings until 31 March 2025 [1] and [3]. |
| 22 | For how long must I keep the supporting documents for my income tax return? | yes | 4 → 4 | According to the passages, you must keep the supporting documents for a period of five (5) years from the date of submission of the return. This applies to all the relevant passages, including [1], [2], [3], and [4]. |
| 23 | What are the source codes on an employee's income tax certificate used for? | yes | 1 → 1 | The source codes on an employee's income tax certificate are used to report the specific type of remuneration paid to an employee, as stated in [2] [2]. The value of the specific remuneration amount must be reported under the relevant code to which that specific type of remuneration relates. |
| 24 | Who is required to file Air Passenger Tax returns and make payments? | yes | 1 → 1 | Every operator liable for Air Passenger Tax (APT) is required to file a return and make payment, in accordance with Section 47B (7) [1]. |
| 25 | Why would SARS ask for access to a company's audit files? | yes | 1 → 1 | SARS may request access to a company's audit files for the purposes of administration of a tax Act, as stated in section 46 of the TAA [4]. This is because the audit files may contain information that is foreseeably relevant for the administration of a tax Act [4]. |
| 26 | What two-factor authentication and password options does eFiling offer? | yes | 1 → 1 | Two-factor authentication methods offered by eFiling include One Time Pin (OTP) [4] and using the SARS MobiApp [1, 3]. Passwordless Login is also an option, but it cannot be used at the same time as Two-Factor Authentication [5]. |
| 27 | What is the annual exclusion for capital gains tax for an individual? | yes | 2 → 3 | The annual exclusion for capital gains tax for an individual is R50 000 for the 2027 year of assessment, but it increases to R440 000 in the year of death [1, 4]. |
| 28 | How are non-executive directors treated for VAT purposes? | no | 1 → 1 | For VAT purposes, non-executive directors (NEDs) are treated as agents, not employees, as confirmed by Binding General Ruling 41 [3]. |
| 29 | What is the standard VAT rate in South Africa? | yes | 1 → 2 | The standard VAT rate in South Africa is 15% [1, 2]. |
| 30 | How does an employer reconcile PAYE using the EMP501 declaration? | no | 1 → 1 | According to [1], an employer should determine the employees' tax to be deducted or withheld from remuneration of each employee for that year and recalculate the employees' tax based on that amount. The employer should also ensure that the IRP5/IT3(a) certificates reflect both taxable and non-taxable source codes, deductions, and the amount of employees' tax deducted from such remuneration. [1] |
| 31 | What is the income tax rate for individuals in Kenya? | yes | (not in the documents) | I don't know based on the uploaded documents. |
| 32 | How much does it cost to renew a South African passport? | yes | (not in the documents) | I don't know based on the uploaded documents. |
| 33 | How do I claim unemployment benefits from the UIF after losing my job? | yes | (not in the documents) | I don't know based on the uploaded documents. |
| 34 | What is the corporate income tax rate in Namibia? | yes | (not in the documents) | I don't know based on the uploaded documents. |
| 35 | When is the deadline for filing a US federal income tax return? | yes | (not in the documents) | I don't know based on the uploaded documents. |
| 36 | What is the fine for driving over the speed limit on a national road? | yes | (not in the documents) | I don't know based on the uploaded documents. |

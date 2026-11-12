# SARS vector database: evaluation

8,317 chunks from 2,671 pages, embedded with bge-small-en-v1.5 (384 dimensions). Both HNSW indexes use 16 links per node, construction width 100 and search width 64.

## Retrieval quality and speed

Quality is measured on 30 hand-written SARS questions. A result counts as correct when it comes from an expected guide and contains the answer phrase. Agreement and speed are measured on 500 further queries drawn from the stored chunks.

| Method | Recall@5 | MRR@10 | Agreement with exact top-10 | Mean query time | 95th percentile |
|---|---|---|---|---|---|
| Exact search (ours) | 0.93 | 0.736 | 1.000 | 0.42 ms | 0.54 ms |
| HNSW (ours) | 0.93 | 0.736 | 0.999 | 1.20 ms | 1.62 ms |
| ChromaDB | 0.93 | 0.736 | 1.000 | 1.82 ms | 2.35 ms |

Turning one question into a vector takes 12.9 ms on average on this CPU (about 7× the slowest search), and is the same for all three methods.

## Build time, memory and disk

| Step | Time |
|---|---|
| Text extraction | 107 s |
| Chunking | 10 s |
| Embedding | 249 s |
| Exact store (vectors + SQLite) | 0.1 s |
| HNSW index (ours, pure Python) | 28 s |
| ChromaDB (HNSW in Rust, plus storage) | 7 s |

- Vector matrix in memory: 12.8 MB
- Our HNSW graph in memory: 9.2 MB for 228,380 links (Python lists; about 0.9 MB as 32-bit integers)
- On disk, ours: 25.9 MB (24.5 MB without the HNSW graph file)
- On disk, ChromaDB: 82.3 MB
- ChromaDB's in-memory size isn't reported: its index lives in native code that Python's memory tracing can't see.

## Per question (rank of first correct result, top 10)

| # | Question | Exact | HNSW | ChromaDB | Top HNSW result |
|---|---|---|---|---|---|
| 1 | What age group must young employees fall in for an employer to claim the Employment Tax Incentive? | 2 | 2 | 2 | SARS Tax Guide for Small Businesses (p. 103) |
| 2 | What is the rate of dividends tax? | 1 | 1 | 1 | Dividends tax (p. 13) |
| 3 | Are gains on shares held as trading stock for a long time taxed as capital or as income? | 2 | 2 | 2 | SARS Tax Guide for Share Owners (p. 7) |
| 4 | When does a foreign supplier of electronic services have to register for VAT in South Africa? | 1 | 1 | 1 | SARS FAQs - Electronic services (p. 17) |
| 5 | What turnover limit must a business stay under to qualify for turnover tax as a micro business? | 1 | 1 | 1 | SARS Tax Guide for Micro Businesses (p. 9) |
| 6 | How long does a taxpayer have to present a case under the mutual agreement procedure of a tax treaty? | 1 | 1 | 1 | LAPD-IT-G24-Guide-on-Mutual-Agreement-Procedures (p. 6) |
| 7 | How much is the monthly medical scheme fees tax credit for the taxpayer? | 7 | 7 | 7 | IT-AE-36-G05 - Comprehensive Guide to the ITR12 Income Tax Return for Individuals - External Guide (p. 94) |
| 8 | What estate duty rate applies to estates worth more than R30 million? | 1 | 1 | 1 | LAPD-IT-G31-FAQs-on-Deceased-Estates (p. 2) |
| 9 | How many days does a taxpayer have to appeal after SARS disallows an objection? | 1 | 1 | 1 | SARS Quick Guide - Alternative Dispute Resolution (p. 5) |
| 10 | What makes a testamentary trust for relatives a type-B special trust? | 2 | 2 | 2 | Part 1 (p. 15) |
| 11 | How is the notional input tax that a motor dealer may deduct on second-hand vehicles limited? | 1 | 1 | 1 | SARS VAT 420 - Guide for Motor Dealers (p. 13) |
| 12 | What counts as an insolvent estate for income tax purposes? | 2 | 2 | 2 | LAPD-IT-G32-FAQs-on-Insolvent-Estates-of-Individuals (p. 5) |
| 13 | When is the first provisional tax payment due? | 3 | 3 | 3 | GEN-PT-01-G01 - Guide for Provisional Tax - External Guide (p. 19) |
| 14 | Since when must a buyer withhold tax when a non-resident sells property in South Africa? | 1 | 1 | 1 | IT-PP-02-G01 - Amounts to be Withheld When a Non-Resident Sells Immovable Property in South Africa(SA) - External Guide (p. 1) |
| 15 | Under which section of the Income Tax Act are public benefit organisations approved? | 3 | 3 | 3 | SARS Guide Guide for Tax Rates/Duties/Levies (p. 9) |
| 16 | By what date must VAT returns be submitted when filing manually compared to eFiling? | 5 | 5 | 5 | GEN-ELEC-04-G01 - Guide to completing the Value Added Tax VAT201 Declaration - External Guide (p. 7) |
| 17 | What is an auto-assessment of an individual's income tax return? | 1 | 1 | 1 | IT-AE-36-G06 - Guide to submit your individual income tax return via eFiling - External Guide (p. 15) |
| 18 | How do I ask SARS to reduce an existing personal income tax assessment? | 1 | 1 | 1 | GEN-GEN-56-G01 - Submit Request for Reduced Assessment RRA01 via eFiling for Individuals - External Guide (p. 1) |
| 19 | How soon must a new employer register for employees' tax, SDL and UIF? | 3 | 3 | 3 | EMP-REG-03-G01 - Guide for completion of Employer Registration application - External Guide (p. 10) |
| 20 | How much transfer duty is payable on the first R1 100 000 of a property's value? | 1 | 1 | 1 | CONTENTS (p. 69) |
| 21 | Until when is the urban development zone allowance available for buildings? | – | – | – | IT-AE-36-G02 - Comprehensive Guide to the Income Tax Return for Trusts - External Guide (p. 38) |
| 22 | For how long must I keep the supporting documents for my income tax return? | 4 | 4 | 4 | IT-GEN-06-G01 - Guide to the Individual ITR12 Return for Deceased and Insolvent Estates - External Guide (p. 6) |
| 23 | What are the source codes on an employee's income tax certificate used for? | 1 | 1 | 1 | PAYE-AE-06-G06 - Guide for Codes Applicable to Employees Tax Certificates 2027 - External Guide (p. 1) |
| 24 | Who is required to file Air Passenger Tax returns and make payments? | 1 | 1 | 1 | APT-03 - Air Passenger Tax - External Guide (p. 1) |
| 25 | Why would SARS ask for access to a company's audit files? | 1 | 1 | 1 | GEN-AU-14-G01 - Access to audit files - External Guide (p. 5) |
| 26 | What two-factor authentication and password options does eFiling offer? | 1 | 1 | 1 | GEN-ELEC-18-G01 - How to Register for eFiling and Manage Your User Profile - External Guide (p. 21) |
| 27 | What is the annual exclusion for capital gains tax for an individual? | 2 | 2 | 2 | SARS Tax Guide for Share Owners (p. 37) |
| 28 | How are non-executive directors treated for VAT purposes? | 1 | 1 | 1 | CHAPTER 1 (p. 9) |
| 29 | What is the standard VAT rate in South Africa? | 1 | 1 | 1 | SARS FAQs - Electronic services (p. 33) |
| 30 | How does an employer reconcile PAYE using the EMP501 declaration? | 1 | 1 | 1 | PAYE-AE-06-G07 - Guide for Validation Rules Applicable to Reconciliation Declarations 2027 - External Guide (p. 4) |

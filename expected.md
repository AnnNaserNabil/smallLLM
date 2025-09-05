t **accuracy above 80%** on  **Bangla + Romanized Bangla dataset**.

Here’s  breakdown:

| Model                                  | Realistic max accuracy on your dataset | Reasoning                                                                                                         |
| -------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **MiniLM-L12-H384-uncased**            | \~70–75%                               | English-only pretraining; may handle Romanized Bangla partially, fails on native Bangla.                          |
| **TinyBERT\_General\_6L\_768D**        | \~65–70%                               | Very small, English-only; likely underfits.                                                                       |
| **albert-base-v2**                     | \~65–70%                               | English-only; not multilingual.                                                                                   |
| **distilbert-base-multilingual-cased** | \~78–82%                               | Multilingual; covers Bangla; fine-tuning on your dataset could push it above 80%.                                 |
| **flaubert\_small\_cased**             | \~50–55%                               | French-focused; minimal Bangla understanding.                                                                     |
| **mobilebert-uncased**                 | \~65–70%                               | English-only; small.                                                                                              |
| **xlm-roberta-comet-small**            | \~80–85%                               | Multilingual; strong coverage of Bangla; small variant is lightweight but can achieve 80+% with good fine-tuning. |

### ⚡ Key Insights

* Only **multilingual models** (DistilBERT-multilingual or XLM-R-comet-small) have a realistic shot at **>80% accuracy**.
* The others are **English-focused** and likely **top out below 75%**.
* Accuracy also depends on **dataset balance, noise, and pre-processing**. Your Banglish (Romanized) data may be tricky for the tokenizer unless it’s multilingual.


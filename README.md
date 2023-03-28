# LT2222 V23 Assignment 3


Part1:
To run the `a3_features.py` script, use the following command:
python3 a3_features.py --folders FOLDER_PATHS --output OUTPUT_FILE --dim DIMENSION
for example: python3 a3_features.py "/Users/jackie/Desktop/data/lt2222-v23/enron_sample" output.csv 100

Part2:
To run the `a3_model.py` script, use the following command:
python3 a3_model.py output.csv

Part3:
To run the model, use the following command:
python3 a3_model.py output.csv --hidden_size HIDDEN_SIZE --activation ACTIVATION
for example: python3 a3_model.py output.csv --hidden_size 50 --activation relu
Test with different hidden layer sizes and activation functions:
| Hidden Layer Size | Activation Function | Accuracy (%) |
|-------------------|---------------------|--------------|
| 50                | None                | 4.78         |
| 50                | ReLU                | 4.78         |
| 50                | Tanh                | 4.78         |
| 100               | None                | 4.78         |
| 100               | ReLU                | 4.78         |
| 100               | Tanh                | 4.78         |

Part4:
The Enron corpus was created as a result of a court subpoena and used to be a significant topic in NLP research. However, the people involved in the corpus never explicitly agreed to have their data used in this manner, and the emails in the corpus were used as evidence against some of them during financial litigation in the collapse of Enron. On the other hand, all the data was provided by them under contract to a company, which can be considered company property, making it difficult to determine whether using the data in this way is reasonable or legal.
First, the privacy of employees is an important ethical consideration. Although the data belonged to the company at the time, using it without explicit permission from the participants may violate their privacy rights. Therefore, when using this data for research, we should take some privacy protection measures. This may include anonymizing the data by removing any information that could expose personal identities, such as names, email addresses, phone numbers, etc., to ensure individuals cannot be identified. Additionally, access to the data can be restricted, allowing only researchers who have been vetted and granted permission to access the data.
Second, the source and legality of the data are also worth considering. While the Enron corpus provides a wealth of resources for NLP research, we should ensure that we follow all applicable laws and regulations to make sure our research is ethical and legal. This may involve understanding the legal provisions related to data collection and processing, as well as following the appropriate data usage agreements.
Lastly, I believe that although the Enron corpus offers a unique research opportunity, researchers should carefully assess the potential negative impacts of their findings and consciously preprocess sensitive information.


import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://www.alibaba.ir/iranout'
response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

details = soup.find_all('details')

questions = []
answers = []

for detail in details:
    question = detail.find(class_='a-accordion__button')
    answer = detail.find(class_='faq-wrapper__description')

    if question and answer:
        questions.append(question.text.strip())
        answers.append(answer.text.strip())

df = pd.DataFrame({'Question': questions, 'Answer': answers})

df.to_excel('questions_and_answers.xlsx', index=False)

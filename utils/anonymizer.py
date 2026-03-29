import re
import json
import os
import requests
from typing import List, Dict, Optional
import pdb


class LegacyAnonymizer:
    def __init__(self, entity_file = './entity.json'):
        self.explicit_ticker_pattern = re.compile(r'(?:\$([A-Z]{1,5}))|(?:[A-Z]+:([A-Z]{1,5}))')
        
        self.tickers: List[str] = []
        self.companies: List[str] = []
        
        if os.path.exists(entity_file):
            try:
                with open(entity_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tickers = data.get('tickers', [])
                    self.companies = data.get('companies', [])
            except Exception as e:
                self.save_tickers_and_companies(entity_file)
        else:
            self.save_tickers_and_companies(entity_file)

    def get_tickers_and_companies(self):
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {
            "User-Agent": "mengyu.wang@ed.ac.uk"
        }
        
        response = requests.get(url, headers=headers)
        data = response.json()
        
        tickers = []
        companies = []
        for key, value in data.items():
            tickers.append(value['ticker'])
            companies.append(value['title'])
            
        return tickers, companies

    def save_tickers_and_companies(self, entity_file):
        print("Downloading tickers and companies")
        tickers, companies = self.get_tickers_and_companies()
        self.tickers = tickers
        self.companies = companies
        
        with open(entity_file, 'w', encoding='utf-8') as f:
            json.dump({
                "tickers": self.tickers,
                "companies": self.companies
            }, f, ensure_ascii=False, indent=2)
        print(f"Sace to {entity_file}")

    def desensitize(self, text, tickers = None, companies = None):
        if not text:
            return text
        tickers = tickers if tickers is not None else self.tickers
        companies = companies if companies is not None else self.companies

        sorted_companies = sorted(list(set(companies)), key=len, reverse=True)
        sorted_tickers = sorted(list(set(tickers)), key=len, reverse=True)

        ticker_map: Dict[str, str] = {}
        company_map: Dict[str, str] = {}
        
        t_counter = 1
        c_counter = 1

        desensitized_text = text

        for company in sorted_companies:
            if company and company in desensitized_text:
                if company not in company_map:
                    company_map[company] = f"[company {c_counter}]"
                    c_counter += 1
                desensitized_text = desensitized_text.replace(company, company_map[company])

        for ticker in sorted_tickers:
            if ticker and ticker in desensitized_text:
                if ticker not in ticker_map:
                    ticker_map[ticker] = f"[ticker {t_counter}]"
                    t_counter += 1
                escaped_ticker = re.escape(ticker)
                
                if re.match(r'^\w+$', ticker):
                    pattern = re.compile(rf'\b{escaped_ticker}\b')
                    desensitized_text = pattern.sub(ticker_map[ticker], desensitized_text)
                else:
                    desensitized_text = desensitized_text.replace(ticker, ticker_map[ticker])

        return desensitized_text



if __name__ == '__main__':
    filter = LegacyAnonymizer()
    text = "Apple Inc. is a technology company that designs, develops, and sells consumer electronics, computer software, and online services. The company's stock ticker is AAPL."
    print(filter.desensitize(text))

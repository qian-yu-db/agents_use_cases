import random
import pandas as pd
from faker import Faker


class FakeDemographicDataGenerator:
    def __init__(self, config, num_records):
        """
        Initialize the generator with the configuration and number of records.

        :param config: A dictionary defining the demographic features to generate.
        :param num_records: Number of records to generate.
        """
        self.config = config
        self.num_records = num_records
        self.fake = Faker()

    def generate_record(self):
        """Generate a single fake demographic record based on the config."""
        record = {}

        for feature, options in self.config.items():
            if feature == "name":
                record[feature] = self.fake.name()
            elif feature == "age":
                record[feature] = random.randint(options["min"], options["max"])
            elif feature == "gender":
                record[feature] = random.choice(options)
            elif feature == "email":
                record[feature] = self.fake.email()
            elif feature == "phone":
                record[feature] = self.fake.phone_number()
            elif feature == "address":
                record[feature] = self.fake.address()
            elif feature == "city":
                record[feature] = self.fake.city()
            elif feature == "country":
                record[feature] = self.fake.country()
            elif feature == "income_level":
                record[feature] = random.choice(options)
            elif feature == "investment_experience":
                record[feature] = random.choice(options)
            elif feature == "risk_aversion":
                record[feature] = random.choice(options)
            elif feature == "investment_preference":
                record[feature] = random.choice(options)

        return record

    def generate_data(self):
        """Generate a pandas DataFrame with fake demographic records."""
        records = [self.generate_record() for _ in range(self.num_records)]
        return pd.DataFrame(records)

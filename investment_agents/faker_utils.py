import random
import pandas as pd
from faker import Faker
from collections import defaultdict


class FakeDemographicDataGenerator:
    def __init__(self, config, num_records):
        """
        Initialize the generator with the configuration and number of records.

        :param config: A dictionary defining the demographic features to generate.
        :param num_records: Number of records to generate.
        """
        self.config = config
        self.num_records = num_records

    def generate(self):
        """Generate a single fake demographic record based on the config."""
        fake = Faker()
        records = defaultdict(list)

        for _ in range(self.num_records):
            for feature, options in self.config.items():
                if options:
                    if isinstance(options, list):
                        records[feature].append(fake.random_element(elements=options))
                    elif isinstance(options, dict):
                        records[feature].append(fake.random_int(min=options['min'], max=options['max']))
                    else:
                        print(f"Invalid options for Faker input {feature}: {options}")
                else:
                    if feature == 'name':
                        records[feature].append(fake.name())
                    elif feature == 'address':
                        records[feature].append(fake.address())
                    elif feature == 'phone_number':
                        records[feature].append(fake.phone_number())
                    elif feature == 'occupations':
                        records[feature].append(fake.job())
                    elif feature == 'city':
                        records[feature].append(fake.city())
                    elif feature == 'country':
                        records[feature].append(fake.country())
        
        return pd.DataFrame(records)


def generate_product_data(products, products_tiers):
    fake = Faker()
    prods = []

    for prod, des in products.items():
        for tier in products_tiers:
            prods.append({
                'product_id': fake.random_int(min=10000, max=99999),
                'product_name': prod,
                'tier': tier,
                'description': des
            })
    return pd.DataFrame(prods)
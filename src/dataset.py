import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

ENTITY_MAPPER = {
    # Location
    "city_name":         "Location",
    "airport_name":      "Location",
    "airport_code":      "Location",
    "state_name":        "Location",
    "state_code":        "Location",
    "country_name":      "Location",

    # Airline
    "airline_name":      "Airline",
    "airline_code":      "Airline",

    # DateTime
    "day_name":          "DateTime",
    "day_number":        "DateTime",
    "month_name":        "DateTime",
    "year":              "DateTime",
    "date_relative":     "DateTime",
    "today_relative":    "DateTime",
    "days_code":         "DateTime",
    "time":              "DateTime",
    "start_time":        "DateTime",
    "end_time":          "DateTime",
    "flight_time":       "DateTime",

    # Modifier
    "period_of_day":     "Modifier",
    "period_mod":        "Modifier",
    "time_relative":     "Modifier",
    "flight_mod":        "Modifier",
    "mod":               "Modifier",

    # Price
    "round_trip":        "Price",
    "cost_relative":     "Price",
    "fare_amount":       "Price",
    "fare_basis_code":   "Price",

    # FlightDetails
    "class_type":        "FlightDetails",
    "transport_type":    "FlightDetails",
    "flight_stop":       "FlightDetails",
    "flight_days":       "FlightDetails",
    "connect":           "FlightDetails",
    "restriction_code":  "FlightDetails",
    "economy":           "FlightDetails",

    # Identifier
    "flight_number":     "Identifier",
    "aircraft_code":     "Identifier",

    # Meal
    "meal":              "Meal",
    "meal_description":  "Meal",
    "meal_code":         "Meal",

    # Other
    "or":                "X",
    "O":                 "X",
}

INTENT_MAPPER = {
    # Booking-related queries (flights, fares, ground service)
    "flight":                      "Flight",
    "airfare":                     "Fare",
    "ground_service":              "Fare",
    "flight+airfare":              "Fare",
    "ground_service+ground_fare":  "Fare",
    "ground_fare":                 "Fare",
    "cheapest":                    "Fare",

    # Flight information (airline, aircraft, flight numbers)
    "airline":                     "FlightInfo",
    "abbreviation":                "FlightInfo",
    "airline+flight_no":           "FlightInfo",
    "aircraft":                    "FlightInfo",
    "flight_no":                   "FlightInfo",
    "aircraft+flight+flight_no":   "FlightInfo",

    # Location-based queries
    "airport":                     "Location",
    "city":                        "Location",
    "distance":                    "Location",

    # Scheduling/time queries
    "flight_time":                 "Schedule",
    "airfare+flight_time":         "Schedule",

    # Capacity/quantity queries
    "capacity":                    "Capacity",
    "quantity":                    "Capacity",

    # Meal-related queries
    "meal":                        "Meal",

    # Restriction or rule queries
    "restriction":                 "Restriction",
}

SLOT2ID = {s: i for i, s in enumerate(set(ENTITY_MAPPER.values()), start=1)}
INTENT2ID = {s: i for i, s in enumerate(set(INTENT_MAPPER.values()))}
pad_tag = "<PAD>"
SLOT2ID[pad_tag] = 0

class ATISDataset(Dataset):
    """
    Loads ATIS-style JSON, tokenizes text, and aligns slot tags & intent labels.
    """
    def __init__(self, data,
                tokenizer: AutoTokenizer, slot2id: dict,
                intent2id: dict, max_len: int = 100):
        self.max_len = max_len
        self.texts = []
        self.slot_tags = []
        self.intent_ids = []
        for d in data:
            if len(d['text']) >= max_len:
                continue 
            self.texts.append(d['text'])
            self.slot_tags.append(d['entity'])
            self.intent_ids.append(intent2id[d['intent']])
            
        self.tokenizer = tokenizer
        self.slot2id = slot2id
        self.intent2id = intent2id
        self.id2intent = {intent: lbl for intent, lbl in intent2id.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.slot_tags[idx]
        intent = self.intent_ids[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)
        
        # Align tags (naive one-tag-per-word); pad/truncate
        tag_ids = [self.slot2id.get(t, self.slot2id["<PAD>"]) for t in tags]
        if len(tag_ids) < self.max_len:
            tag_ids += [self.slot2id["<PAD>"]] * (self.max_len - len(tag_ids))
        else:
            tag_ids = tag_ids[: self.max_len]
            
        slot_labels = torch.tensor(tag_ids, dtype=torch.long)
        intent_label = torch.tensor(intent, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "intent_label": intent_label,
            "slot_labels": slot_labels
        }
        

def load_dataset(data_path, tokenizer_name="bert-base-uncased"):
    train_path = os.path.join(data_path, "train.json")
    test_path  = os.path.join(data_path, "test.json")
    
    with open(train_path, 'r') as f:
        train_records = json.load(f)

    with open(test_path, 'r') as f:
        test_records = json.load(f)
    
    tokenizer     = AutoTokenizer.from_pretrained(tokenizer_name)
    train_dataset = ATISDataset(train_records, tokenizer, SLOT2ID, INTENT2ID)
    test_dataset  = ATISDataset(test_records, tokenizer, SLOT2ID, INTENT2ID)
    
    return train_dataset, test_dataset

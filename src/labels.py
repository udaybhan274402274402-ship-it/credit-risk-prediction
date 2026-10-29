"""Human-readable names for the German Credit codes (see data/raw/german.doc)."""

FEATURE_TITLES = {
    "checking_account_status": "Checking account status",
    "credit_history": "Credit history",
    "purpose": "Purpose of the loan",
    "savings_account": "Savings account / bonds",
    "employment_since": "Present employment since",
    "other_debtors": "Other debtors / guarantors",
    "property": "Property",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing",
    "job": "Job type",
    "telephone": "Telephone",
}

VALUE_LABELS = {
    "checking_account_status": {
        "A11": "Less than 0 DM",
        "A12": "0 to 200 DM",
        "A13": "200 DM or more",
        "A14": "No checking account",
    },
    "credit_history": {
        "A30": "No credits taken / all paid back duly",
        "A31": "All credits at this bank paid back duly",
        "A32": "Existing credits paid back duly till now",
        "A33": "Delay in paying off in the past",
        "A34": "Critical account / other credits elsewhere",
    },
    "purpose": {
        "A40": "Car (new)", "A41": "Car (used)", "A42": "Furniture / equipment",
        "A43": "Radio / television", "A44": "Domestic appliances", "A45": "Repairs",
        "A46": "Education", "A48": "Retraining", "A49": "Business", "A410": "Other",
    },
    "savings_account": {
        "A61": "Less than 100 DM", "A62": "100 to 500 DM", "A63": "500 to 1000 DM",
        "A64": "1000 DM or more", "A65": "Unknown / no savings account",
    },
    "employment_since": {
        "A71": "Unemployed", "A72": "Less than 1 year", "A73": "1 to 4 years",
        "A74": "4 to 7 years", "A75": "7 years or more",
    },
    "other_debtors": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    "property": {
        "A121": "Real estate",
        "A122": "Building savings / life insurance",
        "A123": "Car or other property",
        "A124": "Unknown / no property",
    },
    "other_installment_plans": {"A141": "Bank", "A142": "Stores", "A143": "None"},
    "housing": {"A151": "Rent", "A152": "Own", "A153": "For free"},
    "job": {
        "A171": "Unemployed / unskilled (non-resident)",
        "A172": "Unskilled (resident)",
        "A173": "Skilled employee / official",
        "A174": "Management / self-employed / highly qualified",
    },
    "telephone": {"A191": "None", "A192": "Yes (registered in customer's name)"},
}


def describe_value(feature: str, code: str) -> str:
    """Return a readable name for a code; fall back to the raw code."""
    return VALUE_LABELS.get(feature, {}).get(code, code)
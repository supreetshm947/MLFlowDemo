from dagster import asset

@asset
def run():
    print("Test pipeline")
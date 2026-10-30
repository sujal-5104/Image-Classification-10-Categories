
import csv
from pathlib import Path

results = [
    ["Cassette Player", "Cassette Player", 99.93, True],
    ["Chainsaw", "Chainsaw", 63.88, True],
    ["French Horn", "French Horn", 100.00, True],
    ["Garbage Truck", "Garbage Truck", 99.99, True],
    ["Gas Pump", "Gas Pump", 99.99, True],
    ["Tench", "Tench", 100.00, True],
    ["Springer", "Springer", 100.00, True],
    ["Parachute", "Parachute", 100.00, True],
    ["Golf Ball", "Golf Ball", 99.97, True],
    ["Church", "Church", 98.71, True]
]

project_dir = Path(__file__).resolve().parent
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

report_path = output_dir / "application_test_results.csv"

with open(report_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Actual Category",
        "Predicted Category",
        "Confidence (%)",
        "Correct"
    ])

    writer.writerows(results)

correct = sum(row[3] for row in results)

print("Final Streamlit testing report saved!")
print("Images tested:", len(results))
print("Correct predictions:", correct)
print("Application sample accuracy:", f"{correct / len(results) * 100:.2f}%")
print("Saved at:", report_path)

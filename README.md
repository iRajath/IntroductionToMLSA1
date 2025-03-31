# IntroductionToMLSA1

## Program

```python
# Program developed by : Abishek Priyan M    Register number: 212224240004

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('FuelConsumption.csv')

# Q1: Scatter plot between Cylinder vs CO2Emission (green)
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions')
plt.title('Cylinders vs CO2 Emissions (Q1)')
plt.show()

# Q2: Compare Cylinder vs CO2 and EngineSize vs CO2 with different colors
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='blue')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions')
plt.title('Cylinders vs CO2 (Q2)')

plt.subplot(1, 2, 2)
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.title('Engine Size vs CO2 (Q2)')
plt.tight_layout()
plt.show()

# Q3: Compare Cylinder, EngineSize, and FuelConsumption_comb vs CO2
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions')
plt.title('Cylinders vs CO2 (Q3)')

plt.subplot(1, 3, 2)
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.title('Engine Size vs CO2 (Q3)')

plt.subplot(1, 3, 3)
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='red')
plt.xlabel('Fuel Consumption Comb')
plt.ylabel('CO2 Emissions')
plt.title('Fuel Consumption vs CO2 (Q3)')
plt.tight_layout()
plt.show()

# Q4: Train model with Cylinder as independent variable
X_cyl = df[['CYLINDERS']]
y = df['CO2EMISSIONS']
model_cyl = LinearRegression()
model_cyl.fit(X_cyl, y)
print("Q4 Model Coefficients (Cylinder):", model_cyl.coef_)

# Q5: Train model with FuelConsumption_comb as independent variable
X_fuel = df[['FUELCONSUMPTION_COMB']]
model_fuel = LinearRegression()
model_fuel.fit(X_fuel, y)
print("Q5 Model Coefficients (Fuel Consumption):", model_fuel.coef_)

# Q6: Evaluate models with different train-test ratios
ratios = [0.7, 0.8, 0.9]
print("\nQ6: Model Accuracies with Different Ratios")

def evaluate_model(X, y, ratios):
    for ratio in ratios:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=42)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        print(f"Ratio {ratio}: R² = {score:.4f}")

print("\nCylinder Model:")
evaluate_model(X_cyl, y, ratios)

print("\nFuel Consumption Model:")
evaluate_model(X_fuel, y, ratios)
```

## Output

Q1<BR>
![Untitled](https://github.com/user-attachments/assets/2dbc632f-2463-4c28-8c36-03dcb615df55)

Q2
![Untitled](https://github.com/user-attachments/assets/ff1b24cf-2137-4881-aad4-22c76ad94a54)

Q3
![Untitled](https://github.com/user-attachments/assets/9b0df84e-bf74-438b-8e4a-eb3874cf0345)

Q4
![image](https://github.com/user-attachments/assets/6b858cb9-b76e-4592-a940-20b3616bbeb3)

Q5
![image](https://github.com/user-attachments/assets/a989d159-e1ea-45b2-8c1d-9cd2dcd56533)

Q6
![image](https://github.com/user-attachments/assets/78e7b299-746d-4edc-8b79-9cb9b57328eb)

## Result
Successfully done

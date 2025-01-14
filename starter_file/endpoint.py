import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://003d3859-a205-4e86-b82b-3ecf7e405f21.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'FC31sRBImFyamhrh3VcbUU1FvBcKVMwa'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
           "age": 75, 
           "anaemia": 0, 
           "creatinine_phosphokinase": 582, 
           "diabetes": 0, 
           "ejection_fraction": 20, 
           "high_blood_pressure": 1, 
           "platelets": 265000, 
           "serum_creatinine": 1.9, 
           "serum_sodium": 130, 
           "sex": 1, 
           "smoking": 0,
           "time": 4
          },
          {
           "age": 53, 
           "anaemia": 0, 
           "creatinine_phosphokinase": 60, 
           "diabetes": 1, 
           "ejection_fraction": 60, 
           "high_blood_pressure": 0, 
           "platelets": 371000, 
           "serum_creatinine": 0.9, 
           "serum_sodium": 130, 
           "sex": 1, 
           "smoking": 0,
           "time": 22
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
print("++++++++++++++++++++++++++++++")
print("Expected result: [true, true], where 'true' means '1' as result in the 'DEATH_EVENT' column")



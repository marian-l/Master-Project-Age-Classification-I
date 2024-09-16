from matplotlib import pyplot as plt

gender_distribution = {
    'male': 59,
    'female': 8,
    'no_information': 33
}

age_distribution = {
    'Unknown': 32,
    '20 - 29': 18,
    '40 - 49': 17,
    '30 - 39': 16,
    '50 - 59': 11
}

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, axs = plt.subplots(1, 2)
axs[0].bar(list(age_distribution.keys()), list(age_distribution.values()))
axs[1].bar(list(gender_distribution.keys()), list(gender_distribution.values()))

plt.xlabel('Age/Gender')
plt.ylabel('Percentage')

plt.show()

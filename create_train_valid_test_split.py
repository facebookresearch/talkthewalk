import random
import json

train_neighborhoods = ['fidi', 'williamsburg', 'eastvillage', 'uppereast']
test_neighborhoods = ['hellskitchen']

boundaries = dict()
boundaries['hellskitchen'] = [3, 3]
boundaries['williamsburg'] = [2, 8]
boundaries['eastvillage'] = [3, 4]
boundaries['fidi'] = [2, 3]
boundaries['uppereast'] = [3, 3]


def get_configurations(neighborhoods):
    configurations = list()
    for neighborhood in neighborhoods:
        for min_x in range(boundaries[neighborhood][0]):
            for min_y in range(boundaries[neighborhood][1]):
                boundary_config = list()
                for i in range(4):
                    for j in range(4):
                        x = min_x + i
                        y = min_y + j

                        config = {'neighborhood': neighborhood,
                                  'target_location': [x, y, 0],
                                  'boundaries': [min_x, min_y, min_x + 4, min_y + 4]}

                        boundary_config.append(config)
                configurations.append(boundary_config)
    return configurations

train_configurations, valid_configurations = list(), list()
configs = get_configurations(train_neighborhoods)
for config in configs:
    if random.random() > 0.25:
        train_configurations.extend(config)
    else:
        valid_configurations.extend(config)

test_configurations = list()
for c in get_configurations(test_neighborhoods):
    test_configurations.extend(c)

print(len(train_configurations), len(valid_configurations), len(test_configurations))

with open('configurations.train.json', 'w') as f:
    json.dump(train_configurations, f)

with open('configurations.valid.json', 'w') as f:
    json.dump(valid_configurations, f)

with open('configurations.test.json', 'w') as f:
    json.dump(test_configurations, f)

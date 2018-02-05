import json

neighborhoods = ['fidi', 'williamsburg', 'eastvillage', 'uppereast', 'hellskitchen']

boundaries = dict()
boundaries['hellskitchen'] = [3, 3]
boundaries['williamsburg'] = [2, 8]
boundaries['eastvillage'] = [3, 4]
boundaries['fidi'] = [2, 3]
boundaries['uppereast'] = [3, 3]


def get_configurations(neighborhoods):
    train_configurations = list()
    valid_configurations = list()
    test_configurations = list()

    for neighborhood in neighborhoods:
        cnt = 0
        for minimum_x in range(boundaries[neighborhood][0]+1):
            for minimum_y in range(boundaries[neighborhood][1]+1):
                min_x = minimum_x*2
                min_y = minimum_y*2

                boundary_config = list()
                for i in range(4):
                    for j in range(4):
                        x = min_x + i
                        y = min_y + j

                        config = {'neighborhood': neighborhood,
                                  'target_location': [x, y, 0],
                                  'boundaries': [min_x, min_y, min_x + 3, min_y + 3]}

                        boundary_config.append(config)

                        if minimum_y == 0 and minimum_x < 2:
                            valid_configurations.append(config)
                            cnt += 1
                        elif minimum_y == boundaries[neighborhood][1] and minimum_x < 2:
                            test_configurations.append(config)
                            cnt += 1
                        elif not (minimum_y >= boundaries[neighborhood][1]-1 and minimum_x < 3):
                            train_configurations.append(config)
                            cnt += 1
        print(neighborhood, cnt/16)
    return train_configurations, valid_configurations, test_configurations

train_configurations, valid_configurations, test_configurations = get_configurations(neighborhoods)

print(len(train_configurations), len(valid_configurations), len(test_configurations))

with open('configurations.train.json', 'w') as f:
    json.dump(train_configurations, f)

with open('configurations.valid.json', 'w') as f:
    json.dump(valid_configurations, f)

with open('configurations.test.json', 'w') as f:
    json.dump(test_configurations, f)

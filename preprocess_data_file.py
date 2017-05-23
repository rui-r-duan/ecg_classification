def preprocess_data_file():
    # remove question mark (missing values) in the raw file
    with open('arrhythmia.data.orig') as inputfile:
        with open('arrhythmia.data', 'w') as outputfile:
            for line in inputfile:
                outputfile.write(line.replace('?', ''))

preprocess_data_file()
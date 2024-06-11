def read_file(path):
    try:
        with open(path,'r') as file:
            data = file.read()
            return data
    except FileNotFoundError:
        print("File Not Found!!")
    except Exception as e:
        print(e)

data = read_file("D:\\Parth Dave\\Python\\file.txt")
if data is not None:
    print(data)


# OUTPUT IF FILE EXISTS
# This is txt file

# OUPUT IF FILE DOES NOT  EXISTS
# File Not Found!!
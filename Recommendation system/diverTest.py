from Recommendation_system.driver import Driver

app = Driver
data_one = "../dataset/data_one.xlsx"
data_two = "../dataset/data_two.xls"
data1 = app.readdata(data_one)
data2 = app.readdata(data_two)
app.run(data1, data2, 5,2000)












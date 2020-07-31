from Recommendation_system.driver import Driver

app = Driver()
data_one = "big0.xlsx"
data_two = "big1.xlsx"
data_three = "big2.xlsx"
data1 = app.readdata(data_one)
data2 = app.readdata(data_two)
data3 = app.readdata(data_three)
app.run(data1, data2,data3, 5,2000)











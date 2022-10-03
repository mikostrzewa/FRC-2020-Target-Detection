import tkinter as tk
import csv

def getInput():
	print("Distance: {}".format(distance.get()))
	print("Angle: {}".format(angle.get()))
	print("Force: {}".format(force.get()))
	with open('data.csv', mode='a+') as file:
	    data_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	    data_writer.writerow([distance.get(), angle.get(), force.get()])



window = tk.Tk()
window.title("RaByte Data Collector")

logo = tk.PhotoImage(file="logo.gif")
logo = logo.subsample(10, 10)

lbl = tk.Label(window, text="Team Rabyte", fg="#9F1899", font=("Courier", 44))
img = tk.Label(window, image=logo)
d_lbl=tk.Label(window, text="Distance: ")
a_lbl=tk.Label(window, text="Angle: ")
f_lbl=tk.Label(window, text="Force")

distance = tk.Entry(window,width=10)
angle = tk.Entry(window,width=10)
force = tk.Entry(window,width=10)
btn = tk.Button(window, text="Add", command=getInput)
lbl.place(x=10, y=10)
img.place(x=350, y=50)
d_lbl.place(x=60, y=80)
a_lbl.place(x=60, y=130)
f_lbl.place(x=60, y=180)
distance.place(x=130,y=80)
angle.place(x=130,y=130)
force.place(x=130,y=180)
btn.place(x=100,y=230)

window.geometry('600x300')

window.mainloop()

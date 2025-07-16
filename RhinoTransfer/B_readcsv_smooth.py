import csv
import rhinoscriptsyntax as rs

f = open('test15result.csv', 'r')
rdr = csv.reader(f)
ptlist = []
i = 0
for line in rdr:
    print(line)
    if i == 0:
        pass
    else:
        ram = i%7
        if ram == 0:
            x = line[1]
            y = line[2]
            z = line[3]
            pt = x, y, z
            #rs.AddPoint(pt)
            ptlist.append(pt)
    i = i + 1
f.close()

rs.AddCurve(ptlist)

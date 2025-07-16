import rhinoscriptsyntax as rs
import csv

# 1. Read timestamps from test212.csv
with open('test15result.csv', 'r') as f:
    rdr = csv.reader(f)
    timestamps = []
    i = 0
    for line in rdr:
        if i != 0:
            timestamps.append(float(line[0]))
        i += 1

# 2. Get start and end timestamps
start_time = timestamps[0]
end_time = timestamps[-1]

# 3. Select curve and divide it
crv = rs.GetObject("choose a curve")
pts = rs.DivideCurveLength(crv, 0.01)

# 4. Calculate timestamp gap
if len(pts) > 1:
    timestamp_gap = (end_time - start_time) / (len(pts) - 1)
else:
    timestamp_gap = 0  # Only one point

# 5. Write CSV
with open("kk.csv", "w") as f_out:
    f_out.write("timestamp,x,y,z,r,p,y\n")
    for idx, pt in enumerate(pts):
        timestamp = start_time + idx * timestamp_gap
        x = int(float(pt[0]) * 100) / 100
        y = int(float(pt[1]) * 100) / 100
        z = int(float(pt[2]) * 100) / 100
        line = "{},{},{},{},-91.5,-7,-91.9\n".format(int(timestamp), x, y, z)
        f_out.write(line)

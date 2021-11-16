def syn_list_profile(path="/home/diwu/Project/UnarySim/app/uBrain/hw/module/", filename="syn_list.csv"):
    outfile = open(path + "syn_out.csv", "w")
    outfile.write("Module, area (um^2),dynamic (uW),leakage (uW)\n")

    file = open(path + filename, "r")
    lines = file.readlines()
    for line in lines:
        dut = line.strip()
        area, leakage, dynamic = profile(
            path = path + "32nm_hvt",
            computing = "",
            prefix = dut + "_flat"
            )
        outfile.write(
                    dut + "," +
                    str(area * 1000000.) + ","+
                    str(dynamic * 1000.) + ","+
                    str(leakage * 1000.) + "\n")
    
    print("All designs done.")
    print("==>>Warning: Please mannually process the combinational design results.<<==")
    outfile.close()
    file.close()


def profile(
    path=None,
    computing=None,
    prefix=None
):
    """
    unit: area (mm^2), leakage (mW), dynamic (mW)
    """
    area_file = open(path + "/" + computing + "/" + prefix + "_area.txt", "r")
    area = area_report(area_file)
    power_file = open(path + "/" + computing + "/" + prefix + "_power.txt", "r")
    leakage, dynamic = power_report(power_file)
    area_file.close()
    power_file.close()
    return area, leakage, dynamic


def power_report(
    file=None
):
    """
    all outputs have the unit of mW
    """
    for entry in file:
        elems = entry.strip().split(' ')
        elems = prune(elems)
        if len(elems) >= 6:
            if elems[0] == "Total" and elems[1] == "Dynamic" and elems[2] == "Power" and elems[3] == "=":
                dynamic = float(elems[4])
                unit = str(elems[5])
                if unit == "nW":
                    dynamic /= 1000000.0
                elif unit == "uW":
                    dynamic /= 1000.0
                elif unit == "mW":
                    dynamic *= 1.0
                else:
                    print("Unknown unit for dynamic power:" + unit)

            if elems[0] == "Cell" and elems[1] == "Leakage" and elems[2] == "Power" and elems[3] == "=":
                leakage = float(elems[4])
                unit = str(elems[5])
                if unit == "nW":
                    leakage /= 1000000.0
                elif unit == "uW":
                    leakage /= 1000.0
                elif unit == "mW":
                    leakage *= 1.0
                else:
                    print("Unknown unit for leakage power:" + unit)

    return leakage, dynamic


def area_report(
    file=None
):
    """
    output has the unit of mm^2
    """
    for entry in file:
        elems = entry.strip().split(' ')
        elems = prune(elems)
        if len(elems) >= 3:
            if str(elems[0]) == "Total" and str(elems[1]) == "cell" and str(elems[2]) == "area:":
                area = float(elems[3])

            if str(elems[0]) == "Total" and str(elems[1]) == "area:":
                if str(elems[2]) != "undefined":
                    if area < float(elems[2]):
                        area = float(elems[2])
                    
    area /= 1000000.0
    return area


def prune(input_list):
    l = []

    for e in input_list:
        e = e.strip() # remove the leading and trailing characters, here space
        if e != '' and e != ' ':
            l.append(e)

    return l


if __name__ == "__main__":
    syn_list_profile(path="/home/diwu/Project/UnarySim/app/uBrain/hw/module/", filename="syn_list.csv")

    
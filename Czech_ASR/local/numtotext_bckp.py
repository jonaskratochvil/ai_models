#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
import re


def jednotky(num1):
    if num1 == "0":
        return ""
    elif num1 == "1":
        return "JEDNA"
    elif num1 == "2":
        return "DVA"
    elif num1 == "3":
        return "TŘI"
    elif num1 == "4":
        return "ČTYŘI"
    elif num1 == "5":
        return "PĚT"
    elif num1 == "6":
        return "ŠEST"
    elif num1 == "7":
        return "SEDM"
    elif num1 == "8":
        return "OSM"
    elif num1 == "9":
        return "DEVĚT"


def desitky(num2):
    if num2 == "0":
        return ""
    elif num2 == "1":
        return "DESET"
    elif num2 == "2":
        return "DVACET"
    elif num2 == "3":
        return "TŘICET"
    elif num2 == "4":
        return "ČTYŘICET"
    elif num2 == "5":
        return "PADESÁT"
    elif num2 == "6":
        return "ŠEDESÁT"
    elif num2 == "7":
        return "SEDMDESÁT"
    elif num2 == "8":
        return "OSMDESÁT"
    elif num2 == "9":
        return "DEVADESÁT"


def stovky(num3):
    if num3 == "0":
        return ""
    elif num3 == "1":
        return "STO"
    elif num3 == "2":
        return "DVĚ STĚ"
    elif num3 == "3":
        return "TŘI STA"
    elif num3 == "4":
        return "ČTYŘI STA"
    elif num3 == "5":
        return "PĚT SET"
    elif num3 == "6":
        return "ŠEST SET"
    elif num3 == "7":
        return "SEDM SET"
    elif num3 == "8":
        return "OSM SET"
    elif num3 == "9":
        return "DEVĚT SET"


def tisice(num4):
    if num4 == "0":
        return ""
    elif num4 == "1":
        return "TISÍC"
    elif num4 == "2":
        return "DVA TISÍCE"
    elif num4 == "3":
        return "TŘI TISÍCE"
    elif num4 == "4":
        return "ČTYŘI TISÍCE"
    elif num4 == "5":
        return "PĚT TISÍC"
    elif num4 == "6":
        return "ŠEST TISÍC"
    elif num4 == "7":
        return "SEDM TISÍC"
    elif num4 == "8":
        return "OSM TISÍC"
    elif num4 == "9":
        return "DEVĚT TISÍC"


def desetitisice(num5):
    if num5 == "0":
        return ""
    elif num5 == "1":
        return "DESET"
    elif num5 == "2":
        return "DVACET"
    elif num5 == "3":
        return "TŘICET"
    elif num5 == "4":
        return "ČTYŘICET"
    elif num5 == "5":
        return "PADESÁT"
    elif num5 == "6":
        return "ŠEDESÁT"
    elif num5 == "7":
        return "SEDMDESÁT"
    elif num5 == "8":
        return "OSMDESÁT"
    elif num5 == "9":
        return "DEVADESÁT"

def jedenact_az_devatenact(word):
    if word == "11":
        return "JEDENÁCT"
    elif word == "12":
        return "DVANÁCT"
    elif word == "13":
        return "TŘINÁCT"
    elif word == "14":
        return "ČTRNÁCT"
    elif word == "15":
        return "PATNÁCT"
    elif word == "16":
        return "ŠESTNÁCT"
    elif word == "17":
        return "SEDMNÁCT"
    elif word == "18":
        return "OSMNÁCT"
    elif word == "19":
        return "DEVATENÁCT"

for sentence in sys.stdin:
    arr_sen = sentence.split()
    for i, word in enumerate(arr_sen):

        if word.isdigit() or (word[:-1].isdigit() and int(word[:-1]) > 50) \
                or (word[:-1].isdigit() and word[-1] != "."):
            if word[:-1].isdigit() and not word[-1].isdigit():
                word = word[:-1]
            # if word.isdigit():
            lenght = len(word)
            # JEDNOTKY
            if lenght == 1:
                if word == "0":
                    arr_sen[i] = "NULA"
                elif word == "1":
                    arr_sen[i] = "JEDNA"
                elif word == "2":
                    arr_sen[i] = "DVA"
                elif word == "3":
                    arr_sen[i] = "TŘI"
                elif word == "4":
                    arr_sen[i] = "ČTYŘI"
                elif word == "5":
                    arr_sen[i] = "PĚT"
                elif word == "6":
                    arr_sen[i] = "ŠEST"
                elif word == "7":
                    arr_sen[i] = "SEDM"
                elif word == "8":
                    arr_sen[i] = "OSM"
                elif word == "9":
                    arr_sen[i] = "DEVĚT"

            # 10 - 19
            if lenght == 2 and word[0] == "1":
                if word == "10":
                    arr_sen[i] = "DESET"
                elif word == "11":
                    arr_sen[i] = "JEDENÁCT"
                elif word == "12":
                    arr_sen[i] = "DVANÁCT"
                elif word == "13":
                    arr_sen[i] = "TŘINÁCT"
                elif word == "14":
                    arr_sen[i] = "ČTRNÁCT"
                elif word == "15":
                    arr_sen[i] = "PATNÁCT"
                elif word == "16":
                    arr_sen[i] = "ŠESTNÁCT"
                elif word == "17":
                    arr_sen[i] = "SEDMNÁCT"
                elif word == "18":
                    arr_sen[i] = "OSMNÁCT"
                elif word == "19":
                    arr_sen[i] = "DEVATENÁCT"

            elif lenght == 2:
                arr_sen[i] = str(desitky(word[0])) + \
                    " " + str(jednotky(word[1]))

            elif lenght == 3:
                if int(word[-2:]) < 11 or int(word[-2:]) > 19:
                    arr_sen[i] = str(stovky(word[0])) + " " + \
                        str(desitky(word[1])) + " " + str(jednotky(word[2]))
                else:
                    arr_sen[i] = str(stovky(word[0])) + " " + str(jedenact_az_devatenact(word[-2:]))

            elif lenght == 4:
                if int(word[-2:]) < 11 or int(word[-2:]) > 19:
                    arr_sen[i] = str(tisice(word[0])) + " " + str(stovky(word[1])) + \
                        " " + str(desitky(word[2])) + " " + str(jednotky(word[3]))
                else:
                    arr_sen[i] = str(tisice(word[0])) + " " + str(stovky(word[1])) + \
                        " " + str(jedenact_az_devatenact(word[-2:]))


            elif lenght == 5:
                if int(word[-2:]) < 11 or int(word[-2:]) > 19:
                    arr_sen[i] = str(desetitisice(word[0])) + " " + str(jednotky(word[1])) + \
                        " TISÍC " + \
                        str(stovky(word[2])) + " " + str(desitky(word[3])) + " " + str(jednotky(word[4]))
                else:
                    arr_sen[i] = str(desetitisice(word[0])) + " " + str(jednotky(word[1])) + \
                        " TISÍC " + \
                        str(stovky(word[2])) + " " + str(jedenact_az_devatenact(word[-2:]))

            elif lenght == 6:
                if int(word[-2:]) < 11 or int(word[-2:]) > 19:
                    arr_sen[i] = str(stovky(word[0])) + " " + str(desitky(word[1])) + \
                        " " + \
                        str(jednotky(word[2])) + " TISÍC " + str(stovky(word[3])) \
                        + " " + str(desitky(word[4])) + " " + str(jednotky(word[5]))
                else:
                    arr_sen[i] = str(stovky(word[0])) + " " + str(desitky(word[1])) + \
                        " " + \
                        str(jednotky(word[2])) + " TISÍC " + str(stovky(word[3])) + \
                        " " + str(jedenact_az_devatenact(word[-2:]))

            elif lenght == 7 and word[0] == "1":
                if int(word[-2:]) < 11 or int(word[-2:]) > 19:
                    arr_sen[i] = "MILION " + str(stovky(word[1])) + " " + str(desitky(word[2])) + \
                        " " + \
                        str(jednotky(word[3])) + " TISÍC " + str(stovky(word[4])) + \
                        " " + str(desitky(word[5])) + " " + str(jednotky(word[6]))
                else:
                    arr_sen[i] = "MILION " + str(stovky(word[1])) + " " + str(desitky(word[2])) + \
                        " " + \
                        str(jednotky(word[3])) + " TISÍC " + str(stovky(word[4])) + \
                        " " +str(jedenact_az_devatenact(word[-2:]))

            elif lenght == 7:
                if int(word[-2:]) < 11 or int(word[-2:]) > 19:
                    arr_sen[i] = str(jednotky(word[0])) + " MILIONŮ " + str(stovky(word[1])) + \
                            " " + str(desitky(word[2])) + \
                        " " + \
                        str(jednotky(word[3])) + " TISÍC " + str(stovky(word[4])) + \
                        " " + str(desitky(word[5])) + " " + str(jednotky(word[6]))
                else:
                    arr_sen[i] = str(jednotky(word[0])) + " MILIONŮ " + str(stovky(word[1])) + \
                            " " + str(desitky(word[2])) + " " + str(jednotky(word[3])) + " TISÍC " + \
                            str(stovky(word[4])) + " " + str(jedenact_az_devatenact(word[-2:]))

            #elif lenght == 2:
            #    arr_sen[i] = str(desitky(word[0])) + \
            #        " " + str(jednotky(word[1]))

            #elif lenght == 3:
            #    arr_sen[i] = str(stovky(word[0])) + " " + \
            #        str(desitky(word[1])) + " " + str(jednotky(word[2]))

            #elif lenght == 4:
            #    arr_sen[i] = str(tisice(word[0])) + " " + str(stovky(word[1])) + \
            #        " " + str(desitky(word[2])) + " " + str(jednotky(word[3]))

            #elif lenght == 5:
            #    arr_sen[i] = str(desetitisice(word[0])) + " " + str(jednotky(word[1])) + \
            #        " TISÍC " + \
            #        str(stovky(word[2])) + " " + str(desitky(word[3])
            #                                         ) + " " + str(jednotky(word[4]))

            #elif lenght == 6:
            #    arr_sen[i] = str(stovky(word[0])) + " " + str(desitky(word[1])) + \
            #        " " + \
            #        str(jednotky(word[2])) + " TISÍC " + str(stovky(word[3])
            #                                                 ) + " " + str(desitky(word[4])) + " " + str(jednotky(word[5]))
            #elif lenght == 7 and word[0] == "1":
            #    arr_sen[i] = "MILION " + str(stovky(word[1])) + " " + str(desitky(word[2])) + \
            #        " " + \
            #        str(jednotky(word[3])) + " TISÍC " + str(stovky(word[4])
            #                                                 ) + " " + str(desitky(word[5])) + " " + str(jednotky(word[6]))
            #elif lenght == 7:
            #    arr_sen[i] = str(jednotky(word[0])) + " MILIONŮ " + str(stovky(word[1])) + " " + str(desitky(word[2])) + \
            #        " " + \
            #        str(jednotky(word[3])) + " TISÍC " + str(stovky(word[4])
            #                                                 ) + " " + str(desitky(word[5])) + " " + str(jednotky(word[6]))
        elif (word[:-1].isdigit() and word[-1] == "."):

            if word[:-1] == "1":
                arr_sen[i] = "PRVNÍ"
            elif word[:-1] == "2":
                arr_sen[i] = "DRUHÝ"
            elif word[:-1] == "3":
                arr_sen[i] = "TŘETÍ"
            elif word[:-1] == "4":
                arr_sen[i] = "ČTVRTÝ"
            elif word[:-1] == "5":
                arr_sen[i] = "PÁTÝ"
            elif word[:-1] == "6":
                arr_sen[i] = "ŠESTÝ"
            elif word[:-1] == "7":
                arr_sen[i] = "SEDMÝ"
            elif word[:-1] == "8":
                arr_sen[i] = "OSMÝ"
            elif word[:-1] == "9":
                arr_sen[i] = "DEVÁTÝ"
            elif word[:-1] == "10":
                arr_sen[i] = "DESÁTÝ"
            elif word[:-1] == "11":
                arr_sen[i] = "JEDENÁCTÝ"
            elif word[:-1] == "12":
                arr_sen[i] = "DVANÁCTÝ"
            elif word[:-1] == "13":
                arr_sen[i] = "TŘINÁCTÝ"
            elif word[:-1] == "14":
                arr_sen[i] = "ČTRNÁCTÝ"
            elif word[:-1] == "15":
                arr_sen[i] = "PATNÁCTÝ"
            elif word[:-1] == "16":
                arr_sen[i] = "ŠESTNÁCTÝ"
            elif word[:-1] == "17":
                arr_sen[i] = "SEDMNÁCTÝ"
            elif word[:-1] == "18":
                arr_sen[i] = "OSMNÁCTÝ"
            elif word[:-1] == "19":
                arr_sen[i] = "DEVATENÁCTÝ"
            elif word[:-1] == "20":
                arr_sen[i] = "DVACÁTÝ"
            elif word[:-1] == "21":
                arr_sen[i] = "DVACÁTÝ PRVNÍ"
            elif word[:-1] == "22":
                arr_sen[i] = "DVACÁTÝ DRUHÝ"
            elif word[:-1] == "23":
                arr_sen[i] = "DVACÁTÝ TŘETÍ"
            elif word[:-1] == "24":
                arr_sen[i] = "DVACÁTÝ ČTVRTÝ"
            elif word[:-1] == "25":
                arr_sen[i] = "DVACÁTÝ PÁTÝ"
            elif word[:-1] == "26":
                arr_sen[i] = "DVACÁTÝ ŠESTÝ"
            elif word[:-1] == "27":
                arr_sen[i] = "DVACÁTÝ SEDMÝ"
            elif word[:-1] == "28":
                arr_sen[i] = "DVACÁTÝ OSMÝ"
            elif word[:-1] == "29":
                arr_sen[i] = "DVACÁTÝ DEVÁTÝ"
            elif word[:-1] == "30":
                arr_sen[i] = "TŘICÁTÝ"
            elif word[:-1] == "31":
                arr_sen[i] = "TŘICÁTÝ PRVNÍ"
            elif word[:-1] == "32":
                arr_sen[i] = "TŘICÁTÝ DRUHÝ"
            elif word[:-1] == "33":
                arr_sen[i] = "TŘICÁTÝ TŘETÍ"
            elif word[:-1] == "34":
                arr_sen[i] = "TŘICÁTÝ ČTVRTÝ"
            elif word[:-1] == "35":
                arr_sen[i] = "TŘICÁTÝ PÁTÝ"
            elif word[:-1] == "36":
                arr_sen[i] = "TŘICÁTÝ ŠESTÝ"
            elif word[:-1] == "37":
                arr_sen[i] = "TŘICÁTÝ SEDMÝ"
            elif word[:-1] == "38":
                arr_sen[i] = "TŘICÁTÝ OSMÝ"
            elif word[:-1] == "39":
                arr_sen[i] = "TŘICÁTÝ DEVÁTÝ"
            elif word[:-1] == "40":
                arr_sen[i] = "ČTYŘICÁTÝ"
            elif word[:-1] == "41":
                arr_sen[i] = "ČTYŘICÁTÝ PRVNÍ"
            elif word[:-1] == "42":
                arr_sen[i] = "ČTYŘICÁTÝ DRUHÝ"
            elif word[:-1] == "43":
                arr_sen[i] = "ČTYŘICÁTÝ TŘETÍ"
            elif word[:-1] == "44":
                arr_sen[i] = "ČTYŘICÁTÝ ČTVRTÝ"
            elif word[:-1] == "45":
                arr_sen[i] = "ČTYŘICÁTÝ PÁTÝ"
            elif word[:-1] == "46":
                arr_sen[i] = "ČTYŘICÁTÝ ŠESTÝ"
            elif word[:-1] == "47":
                arr_sen[i] = "ČTYŘICÁTÝ SEDMÝ"
            elif word[:-1] == "48":
                arr_sen[i] = "ČTYŘICÁTÝ OSMÝ"
            elif word[:-1] == "49":
                arr_sen[i] = "ČTYŘICÁTÝ DEVÁTÝ"
            elif word[:-1] == "50":
                arr_sen[i] = "PADESÁTÝ"
        else:
            continue
    # .encode('utf-8').strip() za .join
    new_sentence = " ".join(arr_sen).strip()
    new_sentence = re.sub(' +', ' ', new_sentence)
print(new_sentence)

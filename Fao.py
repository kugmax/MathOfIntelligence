import csv
from enum import Enum


class FaoColumnIndex(Enum):
    AREA_SYM = 0
    AREA_CODE = 1
    AREA_NAME = 2
    ITEM_CODE = 3
    ITEM_NAME = 4
    ELEMENT_CODE = 5
    ELEMENT_NAME = 6
    UNIT = 7
    LATITUDE = 8
    LONGITUDE = 9


class FaoYearsColumnInfo(Enum):
    BEGIN_INDEX = 10
    COUNT = 52
    FIRST_YEAR = 1961
    LAST_YEAR = 2013


class Fao:
    def __init__(self, path):
        self.__path = path

    def get_all_country_supply(self, country='ALL', product=None, element_name=None):
        result = [[]]
        i = 0
        for row in self.__read_data_row():

            if country != 'ALL' and country != row[FaoColumnIndex.AREA_SYM]:
                continue

            if product is not None and product != int(row[FaoColumnIndex.ITEM_CODE]):
                continue

            if element_name is not None and element_name != row[FaoColumnIndex.ELEMENT_NAME]:
                continue

            #print(row)
            #print(len(row))
            #print(type(row))
            result.append([])
            result[i].extend(row)
            i += 1
        return result

    def __read_head_row(self):
        return self.__read_row().next()

    def __read_data_row(self):
        row_generator = self.__read_row()
        row_generator.next()

        for data in row_generator:
            yield data

    def __read_row(self):
        with open(self.__path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                yield row

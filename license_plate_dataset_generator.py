import requests
import re
import random
import time
import numpy as np
import cv2
import os.path
import json
import argparse


class LicensePlateDatasetGenerator:
    def __init__(self, output):
        self.output = output
        self.county_marks = json.loads(open('county_marks.json').read())
        self.charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ"
        self.generator_webservice_url = 'http://nummernschild.heisnbrg.net/fe/task?action=startTask&kennzeichen=%s&kennzeichenZeile2=&engschrift=false&pixelHoehe=100&breiteInMM=520&breiteInMMFest=false&sonder=FE&dd=01&mm=01&yy=00&kreis=LEER_GRAU&kreisName=&humm=08&huyy=17&sonderKreis=LEER&mm1=01&mm2=01&farbe=SCHWARZ&effekt=KEIN&tgaDownload=false'
        random.seed()

    def _generate_license_number(self):
        country_mark_index = random.randint(0, len(self.county_marks) - 1)
        license_number = self.county_marks[country_mark_index]["LKZ"]
        license_number += "-"

        ident_char_len = random.randint(1, 2)
        ident_number_len = random.randint(5, 8)

        for n in range(ident_char_len):
            license_number += self.charset[random.randint(0, len(self.charset) - 1)]

        while len(license_number) < (ident_number_len + 1):
            license_number += str(random.randint(0, 9))

        return license_number

    @staticmethod
    def _get_image_size(image_string):
        image = np.fromstring(image_string, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image.shape[:2]

    def _create_license_plate_picture(self, license_number):
        file_path = self.output + '/%s.png' % license_number
        if os.path.exists(file_path):
            return False

        create_image_url = self.generator_webservice_url % license_number.replace("-", "%3A").replace("Ä", "%C4") \
            .replace("Ö", "%D6").replace("Ü", "%DC")
        r = requests.get(create_image_url)
        if r.status_code != 200:
            return False

        id = re.compile('<id>(.*?)</id>', re.DOTALL | re.IGNORECASE).findall(
            r.content.decode("utf-8"))[0]
        print(id)
        status_url = 'http://nummernschild.heisnbrg.net/fe/task?action=status&id=%s' % id
        time.sleep(.200)
        r = requests.get(status_url)
        if r.status_code != 200:
            return False

        show_image_url = 'http://nummernschild.heisnbrg.net/fe/task?action=showInPage&id=%s'
        show_image_url = show_image_url % id
        time.sleep(.200)
        r = requests.get(show_image_url)
        if r.status_code != 200:
            return False

        # sometimes the web service returns a corrupted image, check the image by getting the size and skip if corrupted
        try:
            self._get_image_size(r.content)
        except:
            return False

        f = open(file_path, 'wb')
        f.write(r.content)
        f.close()
        return True

    def create(self, items):
        for n in range(items):
            while True:
                license_number = self._generate_license_number()
                if self._create_license_plate_picture(license_number):
                    time.sleep(.200)
                    break
                time.sleep(.200)


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--outputdir",
                default="dataset/validation",
                help="output directory")
ap.add_argument("-i", "--items",
                default="100",
                help="number of items to generate")
args = vars(ap.parse_args())

lpdg = LicensePlateDatasetGenerator(args["outputdir"])
lpdg.create(int(args["items"]))

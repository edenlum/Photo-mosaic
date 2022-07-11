from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import json
from collections import namedtuple
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

def click(driver, xpath, delay=10):
    try:
        myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.XPATH, xpath)))
        button = driver.find_element_by_xpath(xpath)
        WebDriverWait(driver, delay).until(button.is_enabled())
        data = button.click()
        print("Page is ready! xpath=", xpath)
    except TimeoutException:
        print("Loading took too much time! xpath=", xpath)


def wait(driver, xpath, delay=10):
    try:
        myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.XPATH, xpath)))
        print("Page is ready! xpath=", xpath)
    except TimeoutException:
        print("Loading took too much time! xpath=", xpath)

def wait_not(driver, xpath, delay=10):
    try:
        EC.element_to_be_clickable
        myElem = WebDriverWait(driver, delay).until_not(EC.presence_of_element_located((By.XPATH, xpath)))
        print("Page is ready! xpath=", xpath)
    except TimeoutException:
        print("Loading took too much time! xpath=", xpath)

def get_driver():
    chrome_options = Options()  
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-notifications")
    s = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=s, options=chrome_options)
    driver.maximize_window()
    return driver

def filter_displayed(li): return list(filter(lambda x: x.is_displayed(), li))

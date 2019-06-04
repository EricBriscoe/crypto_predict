import configparser


def create_config():
    config = configparser.ConfigParser()
    # Config values to write
    api_key = input("Input your API key:\n")
    secret_key = input("Input your secret key:\n")
    config["Binance"] = {"api_key": api_key, "secret_key": secret_key}
    # Save config file
    with open("config.ini", "w") as configfile:
        config.write(configfile)


def retrieve_config_value(category, key):
    config = configparser.ConfigParser()
    config.read("config.ini")
    try:
        value = config[category][key]
    except KeyError:
        create_config()
        config.read("config.ini")
        value = config[category][key]
    return value


if __name__ == "__main__":
    create_config()

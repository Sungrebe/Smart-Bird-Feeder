from create_database import establish_database_conn, get_images

def main():
    col = establish_database_conn()
    get_images()

if __name__ == "__main__":
    main()
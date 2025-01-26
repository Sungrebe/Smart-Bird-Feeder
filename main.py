from create_database import establish_database_conn, upload_images

def main():
    col = establish_database_conn()
    upload_images()

if __name__ == "__main__":
    main()
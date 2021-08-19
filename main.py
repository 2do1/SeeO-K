import extract_clothes

if __name__ == '__main__':
    color = extract_clothes.find_color_name()
    result_list = extract_clothes.find_matching_color_name(color)
    print(result_list)

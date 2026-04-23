from IPython.display import Image, display, HTML

def show_images(image_list, captions):
    display_html = "<table>"
    for img, cap in zip(image_list, captions):
        display_html += f"""
        <tr>
            <td style='text-align:center; padding:10px;'>
                <img src="{img}" width="450"><br>
                <b>{cap}</b>
            </td>
        </tr>
        """
    display_html += "</table>"
    display(HTML(display_html))

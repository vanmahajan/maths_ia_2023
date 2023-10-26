# Author: Angelina van Mahajan
# Calculating area of irregular shapes from a photograph.
#
# convert_selected_to_red
# count_colored_pixels
# riemann_sum
# monte_carlo_multiple
# monte_carlo
# crop_image
# process

from PIL import Image
import matplotlib.pyplot as plt
import random
import statistics


def convert_selected_to_red(image_path, new_path):
    # Open the image using PIL
    image = Image.open(image_path)

    # Convert the image to RGB mode (in case it's not)
    image = image.convert("RGBA")

    # Get the width and height of the image
    width, height = image.size

    # Create a new image to store the modified pixels
    new_image = Image.new("RGBA", (width, height))

    for x in range(width):
        for y in range(height):
            r, g, b, a = image.getpixel((x, y))

            # Check if the pixel is reddish based on the given criteria

            if a > 0.5:
                # Check if the pixel is reddish based on the given criteria
                if (r - g) > 12 and (r - b) > 12:
                    # Convert reddish pixel to blue
                    new_image.putpixel((x, y), (255, 0, 0))
                else:
                    # Preserve the original pixel
                    new_image.putpixel((x, y), (r, g, b, a))
            else:
                new_image.putpixel((x, y), (0, 0, 0, 0))
    new_image.save(new_path)
    return new_image


def count_colored_pixels(image, title, plot_file):
    # Get the width and height of the image
    width, height = image.size

    # Initialize a list to store the count of reddish pixels for each vertical line
    colored_count = []
    print(f"{title}: width={range(width).stop} height={range(height).stop}")
    total_area = 0
    total_door = 0
    colored_area = 0
    for x in range(width):
        count = 0
        for y in range(height):
            r, _, _, a = image.getpixel((x, y))
            total_area += 1
            if a >= 0.5:
                total_door += 1

                # Check if the pixel is reddish based on the given criteria
                if r == 255:
                    count += 1
                    colored_area += 1

        colored_count.append(count)

    print(f"  Colored area: {colored_area}  Door area: {total_door} ")
    plot_sample = colored_count
    plt.figure(figsize=(10, 6))
    plt.plot(
        plot_sample, marker=" ", linestyle="-", label="Function value", color="blue"
    )
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("Function value")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(plot_file)

    return colored_count


def riemann_sum(fn, interval_size, image_prefix):
    # Returns the Riemann Sum where each interval is "interval_size" pixels.
    # fn is value for each pixel
    fnvalues = []
    partial_sum = 0
    plt.figure(figsize=(10, 6))
    x = range(len(fn))
    plt.plot(x, fn, marker=" ", linestyle="-", label="Function value", color="blue")
    for i in range(0, len(fn), interval_size):
        # get a random point in this interval to get the value of the function.
        k = random.randint(0, interval_size - 1)
        if i + k > (len(fn) - 1):
            # Take left point of interval if i+k is out of bounds.
            fnvalue = fn[i]
        else:
            fnvalue = fn[i + k]

        # Using the mean - will get the correct value always
        # fnvalue = statistics.mean(fn[i : i + interval_size])

        fnvalues.append(fnvalue)
        plt.fill_between(
            x[i : i + interval_size + 1], 0, fnvalue, color="gray", alpha=0.4
        )
        partial_sum += fnvalue * interval_size
    print(
        f"  Riemann Sum: Interval size: {interval_size}. Area estimate:{partial_sum:.2f}"
    )
    # print(f"  {fnvalues}")
    plt.title(
        f"Function values with Riemann intervals of {interval_size}. Area estimate={partial_sum:.2f}"
    )
    plt.xlabel("x")
    plt.ylabel("Function value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{image_prefix}_{interval_size}.png")
    return partial_sum


def monte_carlo_multiple(image, iterations, num_points, monte_carlo_file):
    # Run Monte Carlo multiple times.
    estimates = [
        monte_carlo(image, num_points, monte_carlo_file, i > 0)
        for i in range(iterations)
    ]
    mean_estimate = statistics.mean(estimates)
    stddev = statistics.pstdev(estimates)
    print(
        f"  Monte Carlo: {iterations:3} iterations of {num_points:6} points. Estimated area: {mean_estimate:.2f} Std.Deviation: {stddev:.2f}"
    )


def monte_carlo(image, num_points, monte_carlo_file, silent):
    # If silent is True - no image is saved and no message is shown.
    # Get the width and height of the image
    width, height = image.size

    total_area = width * height
    inside_colored_area = 0
    outside_colored_area = 0
    if silent != True:
        new_image = Image.new("RGBA", (width, height))
        for x in range(width):
            for y in range(height):
                new_image.putpixel((x, y), image.getpixel((x, y)))

    for _ in range(num_points):
        # to store the count of reddish pixels for each vertical line
        (x, y) = (random.randint(0, width - 1), random.randint(0, height - 1))
        r, _, _, _ = image.getpixel((x, y))
        if silent != True:
            new_image.putpixel((x, y), (0, 0, 0, 0))
        if r == 255:
            inside_colored_area += 1
        else:
            outside_colored_area += 1

    inside_ratio = (inside_colored_area * 1.0) / (num_points * 1.0)
    estimated_area = inside_ratio * total_area
    if silent != True:
        print(
            f"  Monte Carlo: {1:3}  iteration of {num_points:6} points. Estimated area: {estimated_area:.2f} (Inside:{inside_colored_area:6} Outside:{outside_colored_area:6} Total area: {total_area} )"
        )
        new_image.save(f"{monte_carlo_file}_{num_points}.png")
    return estimated_area


def crop_image(image_path_new, output_path_new):
    # Remove transparent sections
    # Load the newly provided image and convert to 'RGBA' mode

    image_new = Image.open(image_path_new).convert("RGBA")

    # Determine the bounding box of the non-transparent region for the new image
    non_empty_pixels_new = [
        (x, y) for x, y in enumerate(image_new.getdata()) if y[3] > 0
    ]
    if not non_empty_pixels_new:
        raise ValueError("No pixels with alpha > 0 found in the image.")

    # Unzipping the X and Y coordinates
    x_coordinates_new, y_coordinates_new = zip(
        *[
            (pixel[0] % image_new.width, pixel[0] // image_new.width)
            for pixel in non_empty_pixels_new
        ]
    )

    bounding_box_new = (
        min(x_coordinates_new),
        min(y_coordinates_new),
        max(x_coordinates_new),
        max(y_coordinates_new),
    )

    # Crop the new image using the bounding box
    cropped_image_new = image_new.crop(bounding_box_new)

    # Saving the cropped image for demonstration
    cropped_image_new.save(output_path_new)


# Process a single door image
def process(prefix, title):
    source_file = f"./{prefix}_gray.png"
    cropped_file = f"./{prefix}_cropped.png"
    colored_file = f"./{prefix}_red.png"
    plot_file = f"{prefix}_plot.png"
    monte_carlo_file = f"{prefix}_monte_carlo"
    crop_image(source_file, cropped_file)
    im = convert_selected_to_red(cropped_file, colored_file)
    fn = count_colored_pixels(im, title, plot_file)
    monte_carlo_multiple(
        image=im, iterations=10, num_points=5000, monte_carlo_file=monte_carlo_file
    )
    monte_carlo_multiple(
        image=im, iterations=10, num_points=10000, monte_carlo_file=monte_carlo_file
    )
    monte_carlo_multiple(
        image=im, iterations=10, num_points=100000, monte_carlo_file=monte_carlo_file
    )
    monte_carlo_multiple(
        image=im, iterations=10, num_points=200000, monte_carlo_file=monte_carlo_file
    )
    monte_carlo_multiple(
        image=im, iterations=100, num_points=200000, monte_carlo_file=monte_carlo_file
    )
    riemann_sum(fn, interval_size=20, image_prefix=f"{prefix}_riemann")
    riemann_sum(fn, interval_size=10, image_prefix=f"{prefix}_riemann")
    riemann_sum(fn, interval_size=5, image_prefix=f"{prefix}_riemann")
    riemann_sum(fn, interval_size=2, image_prefix=f"{prefix}_riemann")
    riemann_sum(fn, interval_size=1, image_prefix=f"{prefix}_riemann")
    return fn


# Process the woman door
process("woman_door", "Female Symbol")
# Process the Man door
process("man_door", "Male Symbol")

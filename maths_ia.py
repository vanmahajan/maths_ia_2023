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
import numpy as np
import math
from itertools import compress


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


def riemann_sum(fn, interval_size, image_prefix, print_details=False):
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
        if print_details:
            print(
                f"    {int(i/interval_size)+1}, {i}, {i+interval_size},{interval_size}, {fnvalue}, {fnvalue*interval_size}"
            )

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
        p = inside_ratio
        N = num_points
        ci = 1.645 * math.sqrt(p * (1 - p) / N) * total_area
        print(
            f"  Monte Carlo: {1:3}  iteration of {num_points:6} points. Estimated area: {estimated_area:.0f} +/- {ci:4.0f} ( Inside:{inside_colored_area:6} Outside:{outside_colored_area:6} Total area: {total_area} )"
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


def calculate_lebesgue_integral(fn_values, num_intervals, print_details=False):
    """
    Calculate the Lebesgue integral of a function represented by an array of values.

    :param fn_values: Array of function values
    :param num_intervals: Number of intervals to partition the range of f(x)
    :return: The approximate Lebesgue integral, partition information
    """
    # Extracting the relevant function values for the integration range
    # start = 0
    # end = len(fn_values)
    # # relevant_fn_values = fn_values[start : end + 1]
    relevant_fn_values = fn_values

    # Step 1: Partition the range of f(x) into intervals
    min_fn_value = np.min(relevant_fn_values)
    max_fn_value = np.max(relevant_fn_values)
    fn_range = max_fn_value - min_fn_value
    interval_length = fn_range / num_intervals
    if print_details:
        print(f"Min:{min_fn_value} Max:{max_fn_value} Range:{fn_range}")
    intervals = [
        (
            min_fn_value + i * interval_length,
            min_fn_value + (i + 1) * interval_length,
        )
        for i in range(num_intervals)
    ]
    # Tweak the last interval to ensure that the max value of the function is included.
    intervals[-1] = (
        min_fn_value + (num_intervals - 1) * interval_length,
        max_fn_value + interval_length,
    )

    # Step 2: Calculate the Lebesgue measure for each interval
    lebesgue_measures = np.zeros(num_intervals)
    for i, (lower, upper) in enumerate(intervals):
        in_interval = (relevant_fn_values >= lower) & (relevant_fn_values < upper)
        lebesgue_measures[i] = np.sum(in_interval)  # / len(relevant_fn_values)

    # Step 3: Calculate the infimum for each interval
    infimums = np.zeros(num_intervals)
    maximums = np.zeros(num_intervals)
    for i, (lower, upper) in enumerate(intervals):
        in_interval = (relevant_fn_values >= lower) & (relevant_fn_values < upper)
        matching_values = list(compress(relevant_fn_values, in_interval))
        if np.any(in_interval):
            infimums[i] = np.min(matching_values)
            maximums[i] = np.max(matching_values)
        else:
            infimums[i] = 0
            maximums[i] = 0
        if print_details:
            print(
                f"{i+1:0}, {lower:.1f}, {upper:.1f}, {lebesgue_measures[i]:0}, {infimums[i]}, {maximums[i]}"
            )

    # Step 4: Sum up the products of the Lebesgue measure and the infimum for each interval
    lebesgue_lower_integral = np.sum(lebesgue_measures * infimums)
    lebesgue_upper_integral = np.sum(lebesgue_measures * maximums)
    print(
        f"    {num_intervals},{lebesgue_lower_integral:.1f},{lebesgue_upper_integral:.1f}"
    )
    return lebesgue_lower_integral, lebesgue_upper_integral, intervals


# Calculating the Lebesgue integral for f(x) from x = 0 to x = 248
# lebesgue_integral, intervals = calculate_lebesgue_integral(fn_values, 10)
# lebesgue_integral, intervals


def plot_lebesgue(fn_values, intervals, image_prefix):
    """
    Plot the function and the Lebesgue intervals.

    :param fn_values: Array of function values
    :param intervals: List of tuples representing the Lebesgue intervals
    :param image_prefix: Image prefix for generate plot
    """
    x_values = range(len(fn_values))
    x_range = range(len(fn_values))
    plt.figure(figsize=(10, 5))

    # Plotting the function
    plt.plot(x_values, fn_values, label="Function $f(x)$", color="blue")

    # Highlighting the range of integration
    start = 0
    end = len(fn_values)
    plt.axvline(x=start, color="green", linestyle="--", label="Integration Range")
    plt.axvline(x=end, color="green", linestyle="--")
    for i in intervals:
        plt.axhline(y=i[0], color="blue", linestyle="-.")
        plt.axhline(y=i[1], color="blue", linestyle="-.")
    # Plot the end of the last interval and give it a label
    plt.axhline(
        y=intervals[-1][1], color="blue", linestyle="-.", label="Lebesgue intervals"
    )
    # plt.fill_between(
    #     x_values[start : end + 1], fn_values[start : end + 1], color="green", alpha=0.1
    # )

    # Shading the Lebesgue intervals
    # for lower, upper in intervals:
    #     in_interval = (fn_values >= lower) & (fn_values < upper)
    #     plt.fill_between(x_values, fn_values, where=in_interval, color="red", alpha=0.2)

    plt.xlabel("x")
    plt.ylabel("$f(x)$")
    plt.title("Function and Lebesgue Intervals")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{image_prefix}_{len(intervals)}.png")


# Plotting the function and the Lebesgue intervals
# plot_lebesgue(fn_values, intervals)


# Process a single door image
def process(prefix, title):
    source_file = f"./{prefix}_gray.png"
    cropped_file = f"./{prefix}_cropped.png"
    colored_file = f"./{prefix}_red.png"
    plot_file = f"{prefix}_plot.png"
    monte_carlo_file = f"{prefix}_monte_carlo"
    lebesgue_file = f"{prefix}_lebesgue"
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
    riemann_sum(
        fn, interval_size=20, image_prefix=f"{prefix}_riemann", print_details=True
    )
    riemann_sum(fn, interval_size=10, image_prefix=f"{prefix}_riemann")
    riemann_sum(fn, interval_size=5, image_prefix=f"{prefix}_riemann")
    riemann_sum(fn, interval_size=2, image_prefix=f"{prefix}_riemann")
    riemann_sum(fn, interval_size=1, image_prefix=f"{prefix}_riemann")

    print("  Lebesgue integral with details")
    _, _, intervals = calculate_lebesgue_integral(fn, 5, False)
    plot_lebesgue(fn, intervals, lebesgue_file)
    print("  Lebesgue integrals")
    _, _, intervals = calculate_lebesgue_integral(fn, 10)
    _, _, intervals = calculate_lebesgue_integral(fn, 20)
    _, _, intervals = calculate_lebesgue_integral(fn, 30)
    _, _, intervals = calculate_lebesgue_integral(fn, 40)
    _, _, intervals = calculate_lebesgue_integral(fn, np.max(fn) + 1)

    return fn


# Process the woman door
process("woman_door", "Female Symbol")
# Process the Man door
process("man_door", "Male Symbol")

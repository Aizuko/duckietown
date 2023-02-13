---
title: Lab 2
entry_number: 2
---

<!-- THIS IS THE OUTLINE, NOT VISIBLE THO KEEP IT FOR REFERENCE

# Inverse Mathematics - Second lab
 + Link to intro image here

### Heartbeat ros package
 - First ros package
 - Read robot name from environment variable
 - Name spaced using the vehicle argument
 + Path to repo

### Camera modification
 - First look at actually using pre-made topics on the bot
 - Remapped namespacing in a funky way through camera_demo.launch
 - Used a callback function
 - Basename of topic **must** be `compressed` or it won't show up in rqt_image_view
 + Submit screenshot of camera
 + Path to repo, explicity mention it's in the repo instead of a screenshot
 + Also submit a screenshot of the code?

### Robot frame stuff math

TODO \()
    + Link straight-line video
    + Path to repo
    + Image with difference in measurements
    - Explain what different speed levels did
    - Explain we used the `/$(arg veh)/wheels_driver_node/wheels_cmd` topic to move
    - Explain how the topic above was found
    + Link video with different speed levels
    - Explain the rotation task (link failed video, it's the only one we have)
    - TODO: explain math about driving duckiebot in the circle
    - Talk about the use of `rospy.on_shutdown()` hooks to cleanly exit
    - Explain how rosbags work
    + Link script and matplotlib quiver plot of pose

## Driving robot through stages

### General architecture

 1. An odometry publisher node that reads in wheel ticks
 2. An odometry driver that makes decisions from the published pose
 3. An LED node working as a service which the driver node communicates to
 + Link to new waddle workspace

### LED service
 - Mention topic being used to publish LEDs
 - Mention self-constructing the message
 - How service was wrapped around this and testing
 - Use of `rospy.on_shutdown()` hooks twice, in the service node and odometry

### Movement
 - Constructed a simulator, by pretty much ripping out parts of the duckietown wheels node
 - Created a PID controller with good dead reckoning which worked perfectly in the simulator
 - ... Didn't even move when tested on the bot. Requires minimum 0.6 speed to turn
 - Continued to scrap pieces of the PID controller, like the duckietown exit notice
 - Turned into a hardcoded procedural program in the end
 - Talk about params.json workflow optimization
 + Link to driving video
 + Insert AR image of distance measure
 - Explain why it's off...

### Putting it all together
 - Pretty much hardcoded, explain the steps
 - Mention shutdown hooks and rosbag recorders
 + Link to uploaded rosbag file, really
 + Display matplotlib quiver of bag file from before
 - Talk about the perfect circle and imperfect square, and how the circle is wrong

-->


# Inverse Mathematics - Second lab
 + Link to intro image here

## Part 1 - ROS basics

For this first section, we'll be referencing code mostly in the
[heartbeat-ros](https://codeberg.org/akemi/duckietown/src/branch/main/lab2/heartbeat-ros)
workspace

### Heartbeat ros package

We kicked things off by reading about ROS's general publisher-subscriber model.
To extend on the tutorial, we made a [small
package](https://codeberg.org/akemi/duckietown/src/branch/main/lab2/heartbeat-ros/packages/heartbeat_package)
which simply sent a log message every ~50s. The publisher node publishes a
custom message. The subscriber node then receives the message and prints it with
loglevel INFO

To avoid hard coding the robot's name, while still using topic names from root,
we used an odd method to get the hostname by [launching a
subprocess](https://codeberg.org/akemi/duckietown/src/commit/392ef3a55c166f4a18ada428a5793feac5ffc613/lab2/heartbeat-ros/packages/heartbeat_package/src/heartbeat_publisher_node.py#L9).
This works and technically is the most correct way to find a hostname, though we
used the launch-file's `arg` approach to pass in the hostname at runtime for all
of the later packages

### Camera modification

Next up was learning how to interact with pre-made topics on the duckiebot. We
created a single node publishing 2 topics and subscribing to a different two.

To print the image size, we subscribed to `camera_info` and set a callback to
print out a loglevel INFO message about the image's dimensions. The callback
also then republishes the string of the image dimensions to a topic called
`published_image_info`.

On a tip from Justin, who mentioned it may be useful in future labs to remove
color from the camera's image, we attempted to republish a monochrome version of
the image from the duckiebot's camera. Doing this was simply a task for opencv.
The camera callback reads in the raw image into a cv2 buffer, decodes it into
cv2's intermediary representation, applies a `COLOR_BRG2GRAY` filter, re-encodes
it and published it to a topic called `published_compressed`.

So far we've only been mentioning the basename of the topic we're publishing,
and that's since we didn't actually publish to any of these topics. For this
package, we tried to keep all our remapping in the
[camera.launch](https://codeberg.org/akemi/duckietown/src/branch/main/lab2/heartbeat-ros/packages/camera_demo_node/launch/camera.launch)
file. Our actually source code publishes to topics like `~published_image_info`,
though that's just a descriptive placeholder for us to then remap in the launch
file. The actual topics we published to were `~/camera_info` and `~/compressed`.
The same method was used for the subscriber paths, taking advantage of the `veh`
argument at runtime to determine the duckiebot's name.

Keeping all the remapping in the launch file was helpful when we were still
learning how to use name spacing properly. Quickly modifying the paths based on
`rqt_graph` was easier when our launch file was dealing with all the paths,
instead of having to constantly modify source code.

Here's a screenshot of the node in `rqt_graph`:

<img
    src="/images/duckietown/lab2/custom_publisher_and_subscriber_allinone.avif"
    alt="Custom camera node in rqt_graph. It publishes two outgoing topics"
/>

Notice how we used the word `compressed` as the basename of our published
modified image's topic name. While it usually doesn't matter how topics are
named, in this case it was very important our topic name ended with
`compressed`. `rqt_image_view` is hardcoded to only consider topics ending with
`compressed`, so publishing the same image to a topic called `modified_image`
would fail to show up in `rqt_image_view`! Thanks to our classmate `KarlHanEdn`
for discovering this and posting it on the discord

Here's a screenshot of our modified topic being published (this one isn't black
and white):

<img
    src="/images/duckietown/lab2/custom_published_image.avif"
    alt="Picture of screen with rqt_image_view streaming our topic"
/>

You can see the [source code
here](https://codeberg.org/akemi/duckietown/src/branch/main/lab2/heartbeat-ros/packages/camera_demo_node),
though the assignment asks for a picture of the source code... so here's a
picture of where the link leads:

<img
    src="/images/duckietown/lab2/lab2_camera_node_code_screenshot.avif"
    alt="Screenshot of 2 Chromium windows displaying source code"
/>

### Robot Kinematics

**What is the relation between your initial robot frame and world frame? How do you transform between them?**
**How do you convert the location and theta at the initial robot frame to the world frame?**

The robot frame is always centered on the robot, so it is given by
$\mathbf{\xi}_R = \begin{bmatrix}
    x_R \\
    y_R \\
    \theta
\end{bmatrix} = \begin{bmatrix}
    0 \\
    0 \\
    \frac{\pi}{2}
\end{bmatrix}$

The initial world frame is given by
$\mathbf{\xi}_I = \begin{bmatrix}
    x_I \\
    y_I \\
    \theta
\end{bmatrix} = \begin{bmatrix}
    0.32 \\
    0.32 \\
    \frac{\pi}{2}
\end{bmatrix}$

To transform the initial world frame to the robot frame is trivial,
keep the angle $\theta$ the same, and $x_R = 0$ and $y_R = 0$. This is equivalent
to this matrix multiplication:

$\mathbf{\xi}_R = \begin{bmatrix}
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & 1
\end{bmatrix}\mathbf{\xi}_I$

To get the initial world frame from the initial robot frame,
we keep the angle $\theta$ the same, and set $x_I = 0.32$ and $y_I = 0.32$.
This is equivalent to this matrix multiplication:

$\mathbf{\xi}_I = \begin{bmatrix}
    1 & 0 & 0.32 \\
    0 & 1 & 0.32 \\
    0 & 0 & 1
\end{bmatrix}\mathbf{\xi}_R$.

We used the following matrix multiplication to transform between the two:

$\mathbf{\dot{\xi}}_I = \mathbf{R}^{-1}\mathbf{\dot{\xi}}_R$
with
$\mathbf{R}^{-1}(\theta) = \begin{bmatrix}
    \cos(\theta) & -\sin(\theta) & 0 \\
    \sin(\theta) & \cos(\theta) & 0 \\
    0 & 0 & 1
\end{bmatrix}$

Then we can update the world frame by integrating the above changes in world frame

$\mathbf{\xi}_I = \mathbf{\xi}_I + \mathbf{\dot{\xi}}_I$

We also must apply the modulo of $2\pi$ to the angle $\theta$ to keep it between
$0$ and $2\pi$.

We note that the equation for getting the change in robot frame is given by

$\mathbf{\dot{\xi}}_R = \begin{bmatrix}
    \frac{d_r + d_l}{2l} \\
    0 \\
    \frac{d_r - d_l}{2l}
\end{bmatrix}$

where $d_r$ and $d_l$ are the integrated displacement traveled by the right and
left wheels and $l$ is the distance between the wheels and the center of the
rotation.

To get the integrated displacements $d_r$ and $d_l$, we use the wheel encoder
ticks formula:

$\Delta \text{ticks} = \text{ticks}_t - \text{ticks}_{t-1}$

$d_r = 2\pi \cdot r \cdot \frac{\Delta \text{ticks}}{\text{resolution}}$

where $r = 0.025$ is the radius of the Duckiebot wheel and
$\text{resolution} = 135$ is the number of ticks in one rotation of the wheel.

**How did you estimate/track the angles your DuckieBot has traveled?**

To update the angle $\theta$ that our DuckieBot has traveled, we used the
equation

$\theta = \theta + \dot{\theta}$
where
$\dot{\theta} = \frac{d_r - d_l}{2l}$

**Can you explain why there is a difference between actual and desired location?**

There are small errors that are mostly due to slippage and momentum.
Since we do dead reckoning, as the DuckieBot moves more the small errors
compound. We note that errors in the angle tend to drastically affect the bot's
x, y position due to the trigonometric functions used in matrix. This causes
the Duckiebot's desired location to drastically different from the actual location.

**Which topic(s) did you use to make the robot move? How did you figure out the topic that could make the motor move?**

For our first wheel-moving demo we published into the `/$(arg
veh)/wheels_driver_node/wheels_cmd` topic. We found this topic with a mix of
`rostopic list`, `rostopic info`, and `rostopic echo` while using the joystick
to figure out which topics published what. Just as with the camera node, we took
advantage of remapping in the launch file with `arg` to let this run on
different duckiebots. We only used the wheel velocities on this topic's message
(`WheelsCmdStamped`) to guide it forward and negative velocities to guide it
backwards. However we mostly scrapped this code, since the node didn't bother
localizing itself. This'd mean if one of the motors spins slower the other, the
bot would be completely off. It even makes it harder to guarantee we traveled
1.25m, since the two wheels had different distances traveled. While there is a
pre-built duckietown topic which gives pose, we instead opted to make a small
modification out our dead reckoning code from section 2 to record the video
linked below.

[Here's a link](https://www.youtube.com/watch?v=IpDTdR3xOlE) to our duckiebot
going forward then back. We were using a pretty high speed for this one, hence
the little jump it does when starting to reverse. We also hypothesized that part
of the problem here lies in how we immediately go in reverse gear, instead of
slowing down to a stop, then going in reverse. We use this slowing-down idea for
section 2, particularly with our turning logic.

We tried several speeds and came to an amusing conclusion. Using very slow
speeds ~0.3 and very high speeds like 1.0 resulted in about a 10-20cm difference
between the starting and final position. However using intermediary speeds like
0.7, which we thought would do best, actually performed the worst, with and
about 20-40cm difference between the starting and ending positions. For the slow
case, the movement are accurate just since the duckiebot doesn't introduce
nearly as much noise from slippage and momentum. However for high speeds, it
seems the slippage and momentum actually balances out! While the duckiebot will
drift longer from momentum, it will also read more traveled distance when its
wheels quickly slip. We were quite surprised about this, since we expected the
higher the speed the higher the noise, though our 3 empirical runs simply don't
support that conclusion

    + Image with difference in measurements
    - Explain the rotation task (link failed video, it's the only one we have)
    - TODO: explain math about driving duckiebot in the circle
    - Talk about the use of `rospy.on_shutdown()` hooks to cleanly exit
    - Explain how rosbags work
    + Link script and matplotlib quiver plot of pose

### Rotation task

The rotation task was pretty much identical to the driving forward task. We used
the same publisher (`/$(arg
veh)/wheels_driver_node/wheels_cmd`), just now we'd make one of the wheels go [in a
negative](https://codeberg.org/akemi/duckietown/src/commit/392ef3a55c166f4a18ada428a5793feac5ffc613/lab2/waddle/packages/odometry_node/src/odometry_driver_node.py#L158)
velocity, while the other is positive. To figure out our turn angle, we used a
pretty [simple
formula](https://codeberg.org/akemi/duckietown/src/commit/392ef3a55c166f4a18ada428a5793feac5ffc613/lab2/waddle/packages/odometry_node/src/odometry_driver_node.py#L174)
to measure the difference in angle. When we forgot the modulo, our bot sometimes
continued spinning forever. As you'll notice in the code, there're no guarantees
that bot will nail its turn on the first try. Especially when we used high
velocity turns, the bot would often miss the target angle and make a complete
rotation once more before aligning up correctly. While this seems inefficient,
when we tried to make it turn in the closest direction, the slippage this
back-forth adjustment introduced so much noise, the duckiebot's angle ended up
being absurdly far off at high speeds.

In general, slow speeds worked much better for turning. Just [look at
this](https://www.youtube.com/watch?v=WniOrK1jwZs) supposed -90deg turn at max
velocity on both wheels. When we tested in the simulator, we used a turning
velocity of 0.3 and that worked perfectly. It was going to work nicely in
practice too... until we actually tried it. Turns out, the duckiebot is so
heavy, it cannot complete a 90deg turn at any velocity under 0.6. On our second
duckiebot, it sometimes managed with 0.5, though both just came to a halt when
we tried 0.4. This requirement for high-speed turns persisted as the largest
problem in getting demo right in section 2.

### Graceful shutdown

This wasn't mentioned in the previous sections, despite being used in the
driving one, though we used a `rospy.on_shutdown()` hook to make sure our nodes
would shutdown correctly. This method takes in a callback function which is
called when rospy is shutdown. Notably for the driving nodes, it was very
important that we stop the wheels before the node exits, otherwise it'd
endlessly spin. However, that just lead us to write a small script to stop the
wheels

```bash
rostopic pub -f my_msg.yml /path/to/topic msg/type
# Actually used example
rostopic pub -f stopstop /csc22927/wheels_driver_node/wheels_cmd duckietown_msg/WheelsCmdStamped
```

Since the duckiebots occasionally still crash the script without calling
shutdown hooks, this method still ended up being a useful emergency stop for us

## Section 2 - Driving robot through stages

### General architecture

For section 2, we used 3 nodes across 2 packages. You can [view the code
here](https://codeberg.org/akemi/duckietown/src/branch/main/lab2/waddle).

The odometry package includes two nodes. The first node called
`odometry_publisher_node` does our kinematics and localization calculations.
Each wheel is represented by its own dictionary. We mostly do calculations off
the `/{hostname}/{left,right}_wheel_encoder_node/ticks` topic, which publishes
the number of total ticks a wheel has accumulated. According to [the
documentation](https://docs.duckietown.org/daffy/duckietown-robotics-development/out/odometry_modeling.html),
this is 135 per full rotation. Since each wheel likely has a non-zero amount of
ticks when our node starts up, we start by [overwriting the number of wheel
ticks](https://codeberg.org/akemi/duckietown/src/commit/392ef3a55c166f4a18ada428a5793feac5ffc613/lab2/waddle/packages/odometry_node/src/odometry_publisher_node.py#L85)
taken on the first callback, then relative changes can be calculated properly

The other package contains the LED node, which provides a service interface.
This is called from the `odometry_driver_node`, which acts as our central
"state" node

 1. An odometry publisher node that reads in wheel ticks
 2. An odometry driver that makes decisions from the published pose
 3. An LED node working as a service which the driver node communicates to
 + Link to new waddle workspace

### LED service

We first implemented the LED node using the standard publisher-subscriber model
that's usually performed in ROS. This was pretty simple, we just construct an
`LEDPattern` message then [publish
it](https://codeberg.org/akemi/duckietown/src/commit/392ef3a55c166f4a18ada428a5793feac5ffc613/lab2/waddle/packages/led_controls/src/led_controller_node.py#L50)
to the `/{hostname}/led_emitter_node/led_pattern` topic. This worked pretty
well, though we did notice in slow-motion footage that the front LEDs end up
switching noticeably faster than the back LEDs. We didn't notice any difference
with our human eyes. This approach also sometimes just went wonky. Roughly 1/4
runs ended up with a heterochromic duckiebot, at least at one of the states.

In retrospect, we realized the problem is likely since we only publish once. We
should have taken advantage of the service-wrapper to set a message on our led
publisher node, then just have it continuously publish whatever the current
message is at a few Hz. That may have turned out duckiebot back to solid-colors
if the first message got poorly read or something.

Now with the publisher-subscriber model in place, we needed to introduce a
service interface to this node. This took [1
line](https://codeberg.org/akemi/duckietown/src/commit/392ef3a55c166f4a18ada428a5793feac5ffc613/lab2/waddle/packages/led_controls/src/led_controller_node.py#L55),
though about an hour of playing around with it since the [official
tutorial](http://wiki.ros.org/ROS/Tutorials/WritingServiceClient%28python%29)
was a bit confusing about mixing the two models together. In the end, it looks
very similar to a subscriber from the server-side and a publisher from the
client-side, though of course the code is blocking. For our specific case, the
code works just as well with the publisher-subscriber model, though [this
community forum
post](https://answers.ros.org/question/11834/when-should-i-use-topics-vs-services-vs-actionlib-actions-vs-dynamic_reconfigure/)
seems generally in support of a service for how we're using it.

We tested our service by having the node call its own service in a while loop
while alterating colors. The bot ended up looking like a Christmas tree, though
we noticed when the node shutdown it'd leave the last color running. To address
this, we read up on [shutdown
hooks](https://wiki.ros.org/rospy/Overview/Initialization%20and%20Shutdown#Registering_shutdown_hooks)
and set one so that when the led controller node shuts down, it [turns off its
lights](https://codeberg.org/akemi/duckietown/src/commit/392ef3a55c166f4a18ada428a5793feac5ffc613/lab2/waddle/packages/led_controls/src/led_controller_node.py#L69).

On an amusing note, the `LEDPattern` message is written in `rgba`. Usually the
`a` means transparency, so Akemi guessed it must mean the brightness in our
context. However it turns out a lower `a` value gives "transparency" in the
sense of white saturation, so we'd get very pale colors. This leaves us
wondering if we can adjust the brightness beyond on-off, though we didn't figure
that out of this assignment.

### Movement

Getting the movement right took by far the most effort in this lab. Before we
continue, look at [how well it ended
up](https://www.youtube.com/watch?v=NFU9NcNew_w)! Except, that was from the
wrong starting position :sob:. We weren't able to get such a lucky run from the
correct starting position, though here's [our best
recording](https://www.youtube.com/watch?v=mHwQ-8XmVzc). Something seemed to
keep tripping it up on that specific starting tile, since it worked so much
better from the middle.

Before even trying to move the bot, Steven Tang decided to make a simulator so
that we could develop the code at home without the duckiebot. He pretty much
took the [duckietown wheel driver
node](https://github.com/duckietown/dt-duckiebot-interface/blob/daffy/packages/wheels_driver/src/wheels_driver_node.py)
then added [tick
publishers](https://codeberg.org/akemi/duckietown/src/commit/392ef3a55c166f4a18ada428a5793feac5ffc613/lab2/heartbeat-ros/packages/mock_wheels_driver/src/mock_wheels_driver_node.py#L56)
for each wheel. He then added a [run
function](https://codeberg.org/akemi/duckietown/src/commit/392ef3a55c166f4a18ada428a5793feac5ffc613/lab2/heartbeat-ros/packages/mock_wheels_driver/src/mock_wheels_driver_node.py#L114)
which simulated the bot actually moving, while updating its wheel velocities
based on what our odometry node is publishing to the
`/{hostname}/wheels_driver_node/wheels_cmd`.

This is really cool, since it let us work out all the math and dead-reckoning
ahead of time. It was working in the simulator perfectly. Then we put it on the
duckietbot and it didn't even move...

As mentioned previously, we found out the duckiebot doesn't have enough power to
turn without setting the wheel velocity magnitude to at least 0.6, at which
point so much noise is added, the calculations done in the simulator ended up
completely off.

We still retained the localization-based turning from before though. Instead of
telling the duckiebot to turn -90deg at the start, we told it to turn to 0deg in
the world frame.

<div>
TODO
how the whole PID controller thing works, math and frame and all. Talk about the
publisher node
</div>

In the end, our code ended up being [very
procedural](https://codeberg.org/akemi/duckietown/src/commit/392ef3a55c166f4a18ada428a5793feac5ffc613/lab2/waddle/packages/odometry_node/src/odometry_driver_node.py#L275).
For each state, we have a different function. In general they start off by using
a service request to change the LED color, then either move or just wait until
the state is considered over. As usually goes for procedural code like this,
manual tuning becomes really important. We needed to tune a lot, which was an
extremely slow process with the provided duckietown docker container. Every time
we changed a single parameter like the speed, the entire catkin build system
would rerun, which took about 2mins.

To accelerate this, we ended up with a very hacky [parameters
file](https://codeberg.org/akemi/duckietown/src/branch/main/lab2/waddle/config/params.json).
This json effectively stores global variables for the workspace. Nodes read this
[file in
directly](https://codeberg.org/akemi/duckietown/src/commit/392ef3a55c166f4a18ada428a5793feac5ffc613/lab2/waddle/packages/odometry_node/src/odometry_driver_node.py#L47)
and adjust their instance parameters. The genius behind this is how the
Dockerfile doesn't have to rebuild the entire workspace, it only needs to [copy
over the updated
param.json](https://codeberg.org/akemi/duckietown/src/commit/392ef3a55c166f4a18ada428a5793feac5ffc613/lab2/waddle/Dockerfile#L79)
file, which reduced the build time to just a few seconds. We do know that ROS
actually has a parameters system built in, which would allow modifying
parameters at runtime too, though this method was quicker and easier for this
lab. We'll be looking to use the native ROS version in the upcoming labs, since
the build time is unlikely to improve.

Our final video [is here](https://www.youtube.com/watch?v=mHwQ-8XmVzc), the same
one from the start of this section. The final distance was 64cm when measured by
AR-ruler. About 62cm when measured by a prehistoric 90cm-stick:

<div><img
    src="/images/duckietown/lab2/lab2_final_position.avif"
    alt="64cm measured distance in an AR-ruler screenshot of an iPhone"
    style="width: 100%; height: 100%"
/></div>

<div><img
    src="/images/duckietown/lab2/lab2_final_dist_ar.avif"
    alt="64cm measured distance in an AR-ruler screenshot of an iPhone"
/></div>

 - Created a PID controller with good dead reckoning which worked perfectly in the simulator
 - Continued to scrap pieces of the PID controller, like the duckietown exit notice
 - Explain why it's off...

### Putting it all together

For the video above, we recorded [this
bag](https://codeberg.org/akemi/duckietown/src/branch/main/lab2/bag-decoder/kinematics.bag)
and plotted it with matplotlib in [this
ipython-notebook](https://codeberg.org/akemi/duckietown/src/branch/main/lab2/bag-decoder/decode.ipynb),
with the resulting image here:

<img
    src="/images/duckietown/lab2/quiver_plot_sparse.avif"
    alt="A quiver plot (end-to-end arrows) of the robot traveling in a square"
/>

This plot is showing what the duckiebot thinks it's doing, which looking at the
video clearly doesn't align with what's actually taking place. A quiver-plot is
nice, since it gives us a pretty good idea of what the pose is. Notably, unlike
scatter-plots, we can see the duckiebot turning on the spot in the 4 corners. We
can also see the duckiebot go backwards from the top left corner down, just like
we coded it up to do.

The initial turn, which in reality was the worst, looks as if it was nailed spot
on. The duckiebot does pick up the slight drift is has when going in a straight
line, which matches the results we saw in our forward-backward test from
section 1.

We found it amusing how far off the results for our circle were though. In the
quiver-plot, the duckiebot completely most of a perfect circle. In reality, it
did about 1.5 circles in an ellipse. This was likely a mix of slippage, the
wheel's wobble, and the angle of the wheel on the duckiebot, which was certainly
not perfectly parallel with the sides. To test out how bad it is, we modified
our program to make the duckies go in an endless circle and launched it on both
our duckiebots. [Here's a video](https://www.youtube.com/shorts/Qf5KefGTbXg).
Even in that short clip, the two robots clearly had their circles change the
amount they overlap from the start to the end, so there's actually quite a bit
of noise during the turn. On Tuesday Feb 6th's lecture, we also learned how
turning is disastrously more noisy than going in the straight line, so the
quiver-plot's disconnection from reality makes sense.

 - Mention shutdown hooks and rosbag recorders

<!-- COMMENTED STUFF STARTS HERE, IGNORE THIS

## Part 1 - Reading and preparation

### Heartbeat package

Out of interest, or rather to make the template repository useful. I made a
simple heartbeat package to indicate the bot is still running. It has two
publishers and two receivers, each in their own separate private namespace,
which publish and receive a message about once per minute

The launch file has a nice way of remapping topics for each namespace. In the
following example, the subscriber is actually listening for the topic
`~chatter`, though our publisher is publishing to `~heartbeat`. This can all be
remapped in the launch file, without needing to change the underlying code

```xml
<remap from="heartbeat_subscriber/chatter" to="heartbeat_publisher/heartbeat" />
<node
    pkg="heartbeat_package"
    type="heartbeat_subscriber_node.py"
    name="heartbeat_subscriber"
    output="screen"/>
```

Launch files have two ways to specific remaps. The one above I feel is much
worse, though might work for multiple nodes at once. I prefer a more explicit
html style:


```xml
<node
    pkg="heartbeat_package"
    type="heartbeat_subscriber_node.py"
    name="heartbeat_subscriber"
    output="screen">
  <remap from="heartbeat_subscriber/chatter" to="heartbeat_publisher/heartbeat" />
</node>
```

### Emergency

`rostopic pub` sends a repeated stream of the same message to the bot. This can
be used for emergency stops

```bash
rostopic pub -f my_msg.yml /path/to/topic msg/type
# Actually used example
rostopic pub -f stopstop /csc22927/wheels_driver_ndoe/wheels_cmd duckietown_msg/WheelsCmdStamped
```

We can also add a shutdown hook to do something similar in the code, though I'm
not sure it'll work if rospy crashes before shutting down properly

```python
def emergency_halt():
    node.publish_speed(0.0)
    rospy.loginfo("Sent emergency stop")

rospy.on_shutdown(emergency_halt)
```

All node types in duckietown can be found
[here](https://github.com/duckietown/dt-ros-commons/blob/59d4b0b1b565408b7238eafd02be673035771ccf/packages/duckietown/include/duckietown/dtros/constants.py#L46)

### ROS Commands

Here are translated commands between ros one and two. First line is ros1 second
line is ros2:

```bash
rostopic list
ros2 topic list

rostopic echo /rossytopic
ros2 topic echo /rossytopic

    # Get an idea of publishing rate
rostopic hz /rossytopic
ros2 topic hz /rossytopic

    # TODO
rosparam TODO
TODO

# Bags from `cd bag_files` ====
TODO
ros2 bag record -o rossy_bag /rossytopic /camera/feed

TODO
ros2 bag info rossy_bag

TODO
ros2 bag play rossy_bag

```

There's also the catkin vs colcon stuff

```bash
catkin_make  # Sort of... like a zeroconf version of catkin build
colcon build --symlink-install  # Simlink flag is optional
```

For messages, look directly into the source code, since there is no
documentation

[For sensor_msg](http://wiki.ros.org/sensor_msgs)
[For rospy classes](https://github.com/ros/ros_comm/tree/noetic-devel/tools/rosbag/src)

### Dockerz

Dts attempts to handle everything for us, though since it's quite buggy, it'd be
nicer to use directly docker instead. Also,

Startup commands

```bash
# On host
docker compose up -d
docker attach $(docker compose ps -q)

# Now we're in the container
cd /media/duckie_time/duckietown/lab2/<workspace>
catkin_make
source devel/setup.bash

# Ready to go. For example...
roslaunch custom_camera_node camera.launch veh:=csc22927
```


## Other references

[This UM course](https://liampaull.ca/ift6757/schedule/index.html) might be
helpful?

[Duckietown drivers](https://github.com/duckietown/dt-duckiebot-interface)

-->

# What is this model?
This is an experimental vision model that generates captions / prompts of an input image, based on a very long and complex structure.
It combines both booru-style tagging (comma separated keyword tags), and longer descriptive texts.

Example:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/644ed23467c9458c913059ff/09mIVsk_qSxazCCX5TDUN.jpeg)
> waterfall, no_humans, outdoors, scenery, tree, lake, rock, river, water, nature, plant, sky, grass, day, island, blue_sky, solo, mountain, forest, a peaceful and natural landscape image of a waterfall with a pond nestled in the woods has been created using digital art techniques. the vibrant green foliage of the trees, lush pink flowers in full bloom, and the sparkling waters ofthe lagoon create a sense of harmony and tranquility that's hard to put into words. the waterfall stands tall, its majestic beauty accentuated by the lush surroundings. towering over the serene pond, it is like a giant gift from nature itself. situated in a tranquil setting, the scene exudes peacefulness and a feeling of serenity. the image features a beautiful tropical landscape with an impressive waterfall, surrounded by rocks and trees. a few leaves can be seen floating on the water. there are also some flowers scattered about, adding color and texture to the environment. flowers are known for their bright colors and delicate petals that add beauty to any setting. placed strategically, they draw attention to where they are displayed, highlighting the natural beauty of this spectacularly designed piece

# Why though?
It's an experiment in longer and more complex descriptions. My goal is to create a mix of keyword tags and descriptions so that both can be used when prompting, and for the prompt to be of high quality.

This model in it's current state does not succeed with that. It needs further training and refinement.

# How to use
1. Create a virtual environment
2. Install the requirements
3. Add images that you want to caption to the /input/ folder
4. Choose the level of quantization you want in the `inference.py` script. 4, 8 or None. 4 is very fast but worse quality. None is slow but higher quality.
5. Run `py inference.py`

It should download the required models to a /models/ directory. It should be around 11gb in total.

You can also use this script with other Paligemma models.

I recommend: [https://huggingface.co/gokaygokay/paligemma-rich-captions](https://huggingface.co/gokaygokay/paligemma-rich-captions)

Huge thanks to [Gökay Aydoğan](https://github.com/gokayfem) for helping me finetune and convert the model!

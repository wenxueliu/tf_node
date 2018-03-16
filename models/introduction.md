

训练

1. inputs (images tensor) : 可以是任意的输入图片，每个有对输入做任何的假设
2. preprocess : builders.preprocessor_builder 提供了大量的对图片进行预处理的方法
3. predict
4. loss
5. outputs (loss tensor) :  输出总是整数，并且范围在 [0, num_classes)

验证

1. inputs (images tensor)
2. preprocess
3. predict
4. postprocess
5. outputs (boxes tensor, scores tensor, classes tensor, num_detections tensor)

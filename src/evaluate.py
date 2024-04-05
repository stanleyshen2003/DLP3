from utils import dice_score
def evaluate(net, data, device):
    # implement the evaluation function here
    score = 0
    for _, sample in enumerate(data):
        image = sample['image'].to(device)
        mask = sample['mask'].to(device)
        trimap = sample['trimap']
        outputs = net(image)
        score += dice_score(outputs, mask)
    return score / len(data)
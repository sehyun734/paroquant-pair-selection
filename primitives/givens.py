from torch import Tensor


def rotate(
    weight: Tensor,
    pairs: Tensor,
    theta: Tensor,
    inverse: bool = False,
) -> Tensor:
    # inverse는 theta 부호와 rotation 순서를 함께 뒤집음.
    order = range(len(pairs) - 1, -1, -1) if inverse else range(len(pairs))

    for index in order:
        first = pairs[index, :, 0]
        second = pairs[index, :, 1]
        cos = theta[index].cos()
        sin = -theta[index].sin() if inverse else theta[index].sin()
        first_columns = weight[:, first]
        second_columns = weight[:, second]

        # first를 제자리에서 바꾸면 second 계산에 섞이므로 사본에 씀.
        weight = weight.clone()
        weight[:, first] = cos * first_columns - sin * second_columns
        weight[:, second] = sin * first_columns + cos * second_columns

    return weight

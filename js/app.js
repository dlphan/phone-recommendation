import axios from 'axios'

const input = document.querySelector('#url')
const submit = document.querySelector('#submit')
const result = document.querySelector('#result-section')
const algorithm = document.querySelector('#algorithm')
const method = document.querySelector('#method')

const PAGE_MAX = 15
let domparser = new DOMParser()

const crawl = async () => {
	let productLink = input.value

	const requests = Array.from(
		{length: PAGE_MAX },
		(_, i) => axios.get(`${productLink}/danh-gia?p=${i+1}`)
	)

	const info = await axios
		.get(productLink)
		.then(rs => {
			let doc = domparser.parseFromString(rs.data, 'text/html')
			const img = doc.querySelector('.picture img')
			const name = doc.querySelector('h1')
			return {
				src: img.src,
				name: name.textContent
			}
		})

	const reviews = await axios
		.all(requests)
		.then(axios.spread((...responses) => {
			const list2D = responses.map(rs => {
				let doc = domparser.parseFromString(rs.data, 'text/html')
				const comments = doc.querySelectorAll('.ratingLst .par i')
				const listComment = []

				comments.forEach( function(element, index) {
					if (element.textContent) listComment.push(element.textContent)
				})
				return listComment
			}).filter(list => list.length > 0)

			const list = [].concat(...list2D)
			return list
	}))
	return {
		reviews,
		src: info.src,
		name: info.name
	}
}

submit.addEventListener('click', async (e) => {
	e.preventDefault()
	const rs = await crawl()

	const reviewResult = 'Positive'

	axios({
		method: 'post',
		url: 'http://192.168.137.1:5000/result',
		data: {
			reviews: rs.reviews,
			algorithm: algorithm.value
		}
	})
		.then(({ data }) => {
			const positive_index = data.positive_index
			const negative_index = data.negative_index

			const resultHtml = `
			<div class="img-product">
				<img src="${rs.src}" alt="">
			</div>
			<div class="result">
				<h2 class="product-name">${rs.name}</h2>
				<ul>
					<li>
						<span>Total Reviews: </span>
						<span class="total-reviews">${rs.reviews.length}</span>
					</li>
					<li>
						<span>Total Positive Reviews: </span>
						<span class="total-positive">${data.total_positive}</span>
					</li>
					<li>
						<span>Total Negative Reviews: </span>
						<span class="total-negative">${data.total_negative}</span>
					</li>
					<li>
						<span>Recommend: </span>
						<span class="recommend">${data.recommend}</span>
					</li>
				</ul>
				<div class="comments">
					<table>
						<tr>
							<th>Positive</th>
							<th>Negative</th>
						</tr>
						<tr>
							<td>${rs.reviews[positive_index[0]].substring(0,130)}...</td>
							<td>${rs.reviews[negative_index[0]].substring(0,130)}...</td>
						</tr>
						<tr>
							<td>${rs.reviews[positive_index[1]].substring(0,130)}...</td>
							<td>${rs.reviews[negative_index[1]].substring(0,130)}...</td>
						</tr>
					</table>
				</div>
				<a href="#reviews-section" class="read-more">Read more reviews</a>
			</div>
			`
			if (method.value === 'link') {
				result.innerHTML = resultHtml
			} else {
				result.innerHTML = `
					<span>This review is ${reviewResult}</span>
				`
			}
		})
})
import React from 'react';
import 'mdb-react-ui-kit/dist/css/mdb.min.css'
import {
	MDBFooter,
	MDBContainer,
	MDBIcon,
	MDBBtn
  } from 'mdb-react-ui-kit';

function Footer(props) {
	return (
		<>
			<div className='footer'>
			<MDBFooter className='bg-light text-center text-white'>
				<MDBContainer className='p-4 pb-0'>
					<section className='mb-4'>
					<MDBBtn
						floating tag='a'
						className='m-1'
						style={{ backgroundColor: '#3b5998' }}
						href='#!'
						role='button'
					>
						<MDBIcon fab icon='facebook-f' />
					</MDBBtn>

					<MDBBtn
						floating tag='a'
						className='m-1'
						style={{ backgroundColor: '#55acee' }}
						href='#!'
						role='button'
					>
						<MDBIcon fab icon='twitter' />
					</MDBBtn>

					<MDBBtn
						floating tag='a'
						className='m-1'
						style={{ backgroundColor: '#dd4b39' }}
						href='#!'
						role='button'
					>
						<MDBIcon fab icon='google' />
					</MDBBtn>
					<MDBBtn
						floating tag='a'
						className='m-1'
						style={{ backgroundColor: '#ac2bac' }}
						href='#!'
						role='button'
					>
						<MDBIcon fab icon='instagram' />
					</MDBBtn>

					<MDBBtn
						floating tag='a'
						className='m-1'
						style={{ backgroundColor: '#0082ca' }}
						href='#!'
						role='button'
					>
						<MDBIcon fab icon='linkedin-in' />
					</MDBBtn>

					<MDBBtn
						floating tag='a'
						className='m-1'
						style={{ backgroundColor: '#333333' }}
						href='#!'
						role='button'
					>
						<MDBIcon fab icon='github' />
					</MDBBtn>
					</section>
				</MDBContainer>

				<div className='text-center p-3' style={{ backgroundColor: 'rgba(0, 0, 0, 0.2)' }}>
					© 2022 Copyright: 
					<a className='text-white' href='#!'>
						 aeCare
					</a>
				</div>
				</MDBFooter>
			</div>
		</>
	);
};

export default Footer;